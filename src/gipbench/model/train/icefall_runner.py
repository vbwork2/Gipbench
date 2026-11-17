"""Chạy icefall zipformer/finetune_cs.py theo đoạn, có watchdog (trước đây là stage 4 của ft/run_e3_v1*.sh).

Giữ đúng luật của bản shell:
  - checkpoint mỗi SEG = ceil(STEPS / segments_per_epoch) step; epoch ảo = EPS = SEG * segments_per_epoch step, với
    STEPS = int(epoch_hours * 3600 / (max_duration * 0.93)) (ước số step / epoch ảo)
  - watchdog mỗi poll_s (60 s): dọn checkpoint (giữ bội của EPS + checkpoint mới nhất); RAM trống < min_ram_mb 2 lần liên
    tiếp (bỏ qua ram_grace_s đầu, lúc nạp checkpoint) hoặc ổ < min_free_disk_gb -> SIGINT tiến trình chính rồi dừng đoạn;
    log đứng >= hang_s sau hang_s đầu -> SIGTERM, chờ 20 s, SIGKILL cả nhóm tiến trình (gồm worker spawn_main)
  - hết đoạn: "Done!" trong 5 dòng cuối log -> xong; ổ < min_free_disk_gb -> dừng hẳn; watchdog dừng vì RAM / treo ->
    resume từ checkpoint mới nhất; lỗi khác 2 lần liên tiếp -> dừng
  - resume: --do-finetune 0 --start-epoch 2 --num-epochs 1 --start-batch N. icefall: resume cần --start-epoch > 1 (assert),
    rồi tự đặt lại start_epoch = cur_epoch đã lưu (= 1). Với --num-epochs 2 thì sau khi hết manifest nó chạy THÊM một lượt
    nữa (epoch 2) -> phải là 1.
Khác bản shell: tín hiệu gửi đúng tiến trình con (và nhóm của nó), không pkill theo tên; số GB trống làm tròn LÊN như
`df -BG`. Hàm đo (RAM, ổ) và thời gian chờ truyền vào được để test bằng tiến trình giả.
"""
import datetime as dt
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return 1 << 30


def disk_free_gb(path):
    return math.ceil(shutil.disk_usage(path).free / 2 ** 30)


def steps_per_epoch(epoch_hours, max_duration):
    return int(epoch_hours * 3600 / (max_duration * 0.93))


def segment_plan(steps, segments_per_epoch=4):
    seg = (steps + segments_per_epoch - 1) // segments_per_epoch
    return seg, seg * segments_per_epoch


def ckpt_nums(exp_dir):
    out = []
    for p in Path(exp_dir).glob("checkpoint-*.pt"):
        m = re.fullmatch(r"checkpoint-(\d+)\.pt", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def cleanup(exp_dir, eps):
    """Giữ checkpoint ở ranh giới epoch ảo (bội của eps) và checkpoint mới nhất, xoá phần còn lại."""
    nums = ckpt_nums(exp_dir)
    if not nums:
        return
    last = nums[-1]
    for n in nums:
        if n != last and n % eps:
            (Path(exp_dir) / f"checkpoint-{n}.pt").unlink(missing_ok=True)


def resume_args(exp_dir, init_ckpt):
    nums = ckpt_nums(exp_dir)
    if not nums:
        return ["--do-finetune", "1", "--finetune-ckpt", str(init_ckpt), "--start-epoch", "1", "--num-epochs", "1"]
    return ["--do-finetune", "0", "--start-epoch", "2", "--num-epochs", "1", "--start-batch", str(nums[-1])]


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Runner:
    def __init__(self, cmd_fn, cwd, env, exp_dir, eps, log, seg_log, watchdog, disk_path,
                 mem_fn=mem_available_mb, disk_fn=disk_free_gb, sleep=time.sleep, kill_wait_s=20, retry_wait_s=20):
        """cmd_fn(resume_args) -> argv của finetune_cs.py. log: finetune.log (ghi nối), seg_log: segments.log."""
        self.cmd_fn, self.cwd, self.env, self.exp_dir, self.eps = cmd_fn, cwd, env, Path(exp_dir), eps
        self.log, self.seg_log = Path(log), Path(seg_log)
        self.wd = dict(watchdog)
        self.disk_path = disk_path
        self.mem_fn, self.disk_fn, self.sleep = mem_fn, disk_fn, sleep
        self.kill_wait_s, self.retry_wait_s = kill_wait_s, retry_wait_s

    def _append(self, path, text):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    def _watch(self, proc, stop, events):
        t0 = time.monotonic()
        low = 0
        min_disk, hang = self.wd.get("min_free_disk_gb", 3), self.wd.get("hang_s", 600)
        while not stop.wait(self.wd.get("poll_s", 60)):
            cleanup(self.exp_dir, self.eps)
            free, mem = self.disk_fn(self.disk_path), self.mem_fn()
            el = time.monotonic() - t0
            low = low + 1 if (mem < self.wd.get("min_ram_mb", 1024) and el >= self.wd.get("ram_grace_s", 180)) else 0
            if free < min_disk or low >= 2:
                kind = "disk" if free < min_disk else "ram"
                events.append((kind, f"{_now()} WATCHDOG: {kind}: ổ còn {free} GB, RAM {mem} MB -> dừng finetune\n"))
                proc.send_signal(signal.SIGINT)
                return
            age = time.time() - self.log.stat().st_mtime if self.log.exists() else 0
            if el >= hang and age >= hang:
                events.append(("hang", f"{_now()} WATCHDOG: treo (log đứng {int(age)} s) -> dừng finetune\n"))
                proc.terminate()
                self.sleep(self.kill_wait_s)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, AttributeError):
                    proc.kill()
                return

    def run(self, init_ckpt, max_segments=1000):
        """Chạy tới khi xong / dừng. Trả về "done" | "disk" | "failed" | "max_segments"."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        fails = 0
        for _ in range(max_segments):
            res = resume_args(self.exp_dir, init_ckpt)
            line = f"{_now()} == đoạn mới: {' '.join(res)}\n"
            print(line, end="")
            self._append(self.log, line)
            self._append(self.seg_log, line)
            with open(self.log, "a", encoding="utf-8") as lf:
                proc = subprocess.Popen(self.cmd_fn(res), cwd=self.cwd, env=self.env, stdout=lf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                stop, events = threading.Event(), []
                th = threading.Thread(target=self._watch, args=(proc, stop, events), daemon=True)
                th.start()
                rc = proc.wait()
                stop.set()
                th.join()
            cleanup(self.exp_dir, self.eps)
            for _, msg in events:
                self._append(self.seg_log, msg)
            self._append(self.seg_log, f"{_now()} đoạn kết thúc rc={rc}\n")
            tail = self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
            if any("Done!" in l for l in tail):
                print("finetune XONG")
                return "done"
            kinds = {k for k, _ in events}
            if "disk" in kinds:
                print(f"DỪNG: ổ còn < {self.wd.get('min_free_disk_gb', 3)} GB")
                return "disk"
            if kinds:  # RAM thấp / treo: resume từ checkpoint mới nhất
                self.sleep(self.retry_wait_s)
                continue
            fails += 1
            print(f"finetune lỗi rc={rc} (lần {fails})")
            if fails >= 2:
                print(f"DỪNG: lỗi 2 lần liên tiếp, xem {self.log}")
                return "failed"
            self.sleep(self.retry_wait_s)
        return "max_segments"
