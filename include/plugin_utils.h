#pragma once

#include <Windows.h>
#include <stdio.h>
#include <stdint.h>

static void init_console()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONIN$", "r", stdin);
}

static int64_t get_time() {
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return li.QuadPart;
}

static void print_time(int64_t prev, const char* name) {
	int64_t now = get_time();
	LARGE_INTEGER li;
	QueryPerformanceFrequency(&li);
	double msec = (double)(now - prev) / li.QuadPart * 1000.0;
	printf("%s %f ms\n", name, msec);
}
