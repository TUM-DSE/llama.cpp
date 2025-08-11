#pragma once

#include <pthread.h>
#include <semaphore.h>

#include <string>

#define SHM_LOCATION "/dev/shm/guardian_shm"
#define SHM_SIZE (32 * 1024 * 1024)
#define MAX_REQUESTS 512
#define MAX_TEXT 8196
#define NB_PRIORITIES 3
#define SHM_NAME "Guardian_shm"
#define BENCHMARK_PORT 0xf4

extern "C" enum Lora { NO_LORA, SQL, FOOD, SIZE };

extern "C" struct Request {
    pthread_mutex_t mutex;
    pthread_mutex_t throughput_mutex;
    sem_t clientNotifier;
    sem_t serverNotifier;
    int id;
    int n_chars_to_gen;
    char text[MAX_TEXT];
    int prio;
    enum Lora lora;
    float throughput;
	int outb_id;

	//results
	float tokens_per_second;
	float kv_cache_util;
	float decode_time_ms;
	float scheduling_time_ms;
};

extern "C" struct SharedMemory {
	pthread_mutex_t test_mutex;
    struct Request requests[MAX_REQUESTS];
    sem_t active_reqs;
	int test_var;
    sem_t run;  // server will run while this is larger than 0
};

extern "C" struct Communication {
    void* requestQueue;
};

extern "C" int init_shm(bool create, Communication* comm);
extern "C" int clean_shm(struct SharedMemory* shm);
extern "C" void test_shm();
extern "C" std::string perform_inference(struct Communication* comm, const char* prompt, int n_chars_to_gen, int prio, Lora lora, int outb_id);
extern "C" void init_guardian();
extern "C" const char* call_guardian(const char* request, const int tokens_to_gen, const int outb_id);
extern "C" void freeme(char* ptr);
extern "C" void my_outl(unsigned int id, unsigned int value);
extern "C" void my_ioperm(unsigned short port_nb);
