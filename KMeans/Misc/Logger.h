#ifndef LOGGER
#define LOGGER

void LogError(const char* message)
{
    printf("\033[1;31m");
    printf("%s\n", message);
    printf("\033[0m");
}

void LogInfo(const char* message)
{
    printf("\033[1;34m");
    printf("%s\n", message);
    printf("\033[0m");
}

void LogPass(const char* message)
{
    printf("\033[1;32m");
    printf("%s\n\n", message);
    printf("\033[0m");
}

#endif