#include "shm.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/io.h>


#include "misc.h"


void my_outl(unsigned int id, unsigned int value)
{
	// only take the last 16 bits of each value;
	int event_id = id & 0xFFFF;
	int event_value = value & 0xFFFF;

	int event = (event_id << 16) | event_value;

	printf("Event id: %x\nEvent value: %x\nEvent: %x\n\n", event_id, event_value, event);

	outl(event, BENCHMARK_PORT);
}

int init_shm(bool create, Communication* comm) {

	printf("Entered function shm_init\n");

    int fd;

	int flags = create ? O_CREAT | O_RDWR : O_RDWR;
    fd = shm_open(SHM_NAME, flags, 0777);
    assert(fd != -1);

    long pagesize = sysconf(_SC_PAGESIZE);
    long shm_size = round_up(sizeof(struct SharedMemory), pagesize);
    if (create && ftruncate(fd, shm_size) == -1) {
        perror("ftruncate");
        return -1;
    }

	printf("Mmapping file\n");

	printf("Fd = %d\n",fd);
	printf("shm_size = %ld\n",shm_size);
	printf("page_size = %ld\n",pagesize);
    comm->requestQueue =
        (void*)mmap(NULL, shm_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, fd, 0);
    assert(comm->requestQueue != NULL);
	if(comm->requestQueue == MAP_FAILED) {
		printf("mmap failed with %s\n", strerror(errno));
	}
    assert(comm->requestQueue != MAP_FAILED);

	printf("Successfully mapped file\n");
	printf("Size mutex: %ld\n", sizeof(pthread_mutex_t));
	printf("Size sem_t: %ld\n", sizeof(sem_t));

    close(fd);

	SharedMemory* shm = (SharedMemory*)comm->requestQueue;

    if (!create) {
		//wait on text mutex
		printf("Communication size: %d\n", sizeof(Communication));
		//text mutex bytes:
		printf("Mutext addr: %p\n", (char*)&shm->test_mutex );

		printf("Taking test mutex\n");
		pthread_mutex_lock(&shm->test_mutex);
		printf("Inside critical section\n");
		pthread_mutex_unlock(&shm->test_mutex);
        // only connect to shared memory, do not set up semaphores and mutextes
		//test mutex bytes:
		printf("[C] Mutex bytes:\n");
		for(int i = 0; i < 40; i++) {
			printf("0x%x\n",((char*)&shm->test_mutex)[i]);
		}


		printf("Shared mem ptr: %p\n", shm);
		return 0;
    }

    memset(shm, 0, shm_size);

    // Init mutexes
	int ret;
    pthread_mutexattr_t attr;
    ret = pthread_mutexattr_init(&attr);
	if(ret != 0) {
		perror("failed to init test mutex attr\n");
		exit(EXIT_FAILURE);
	}

	ret = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
	if(ret != 0) {
		perror("failed to set test mutex attr\n");
		exit(EXIT_FAILURE);
	}

	ret = pthread_mutex_init(&shm->test_mutex, &attr);
	if(ret != 0) {
		perror("failed to init test mutex\n");
		exit(EXIT_FAILURE);
	}

	//test mutex bytes:
	printf("[C] Mutex bytes:\n");
	for(int i = 0; i < 40; i++) {
		printf("0x%x\n",((char*)&shm->test_mutex)[i]);
	}
	printf("\n");
	fflush(stdout);

    sem_destroy(&shm->active_reqs);
    if (sem_init(&shm->active_reqs, 1, 0) != 0) {
        perror("sem_init failed");
        exit(EXIT_FAILURE);
    }

    sem_destroy(&shm->run);
    if (sem_init(&shm->run, 1, 1) != 0) {  // init to 1 since server will start
        perror("sem_init failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < MAX_REQUESTS; i++) {
        pthread_mutex_init(&shm->requests[i].mutex, &attr);
        sem_destroy(&shm->requests[i].clientNotifier);
        if (sem_init(&shm->requests[i].clientNotifier, 1, 0) != 0) {
            perror("sem_init failed");
            exit(EXIT_FAILURE);
        }
        sem_destroy(&shm->requests[i].serverNotifier);
        if (sem_init(&shm->requests[i].serverNotifier, 1, 0) != 0) {
            perror("sem_init failed");
            exit(EXIT_FAILURE);
        }
    }


	shm->requests[0].prio = 123;
	shm->requests[1].prio = 124;
	shm->requests[2].prio = 125;
	shm->requests[5].prio = 126;
	shm->requests[10].prio = 127;
	shm->test_var = 321;
	printf("Share requests pointer: %p\n", shm);

	printf("Testing mutex\n");
    pthread_mutex_lock(&shm->requests[0].mutex);
    pthread_mutex_unlock(&shm->requests[0].mutex);

	printf("Shared memory initialized successfully\n");
    return 0;
}

int clean_shm(struct SharedMemory *shm) {
    munmap(shm, SHM_SIZE);
    return 0;
}

void test_shm()
{
	printf("Hello Guardian\n");
}
