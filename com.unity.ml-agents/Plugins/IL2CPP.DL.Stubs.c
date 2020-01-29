// These stubs fix an issue compiling GRPC on Windows with IL2CPP.

void * dlopen(const char *filename, int flags) {
    return 0;
}

void * dlsym(void *handle, const char *symbol) {
    return 0;
}