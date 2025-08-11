#pragma once

inline long round_up(long n, long mult) {
    return ((n + mult - 1) / mult) * mult;
}
