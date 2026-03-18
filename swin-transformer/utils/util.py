import os
import sys
import time

try:
    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)
except Exception as e:
    print(f"Error occurred while getting terminal width: {e}")
    term_width = 80

TOTAL_BAR_LENGTH = 65
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    fraction = (current + 1) / total
    cur_len = int(TOTAL_BAR_LENGTH * fraction)
    cur_len = min(cur_len, TOTAL_BAR_LENGTH)

    bar = ["." for _ in range(TOTAL_BAR_LENGTH)]

    for i in range(cur_len):
        bar[i] = "="

    if cur_len < TOTAL_BAR_LENGTH:
        bar[cur_len] = ">"

    progress_str = f" {current + 1}/{total} "

    start_pos = (TOTAL_BAR_LENGTH - len(progress_str)) // 2
    for i, char in enumerate(progress_str):
        if start_pos + i < TOTAL_BAR_LENGTH:
            bar[start_pos + i] = char

    bar_str = "".join(bar)

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    time_str = f" Step: {format_time(step_time)} | Tot: {format_time(tot_time)}"
    if msg:
        time_str += f" | {msg}"

    line = f" [{bar_str}] {time_str}"
    padding = " " * max(0, term_width - len(line) - 1)

    sys.stdout.write("\r" + line + padding)
    sys.stdout.flush()

    if current >= total - 1:
        sys.stdout.write("\n")


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f
