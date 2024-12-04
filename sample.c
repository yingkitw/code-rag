#include <stdio.h>
#include "sample.h"

int main() {
    printf("Hello, %s!\n", get_name());
    printf("Magic number: %d\n", MAGIC_NUMBER);
    return 0;
}