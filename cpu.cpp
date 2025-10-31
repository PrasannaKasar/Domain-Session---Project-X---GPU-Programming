#include <chrono>

/* Only needed for the sake of this example. */
#include <iostream>
#include <thread>
#include <vector>
    
void long_operation(int* arr, int size)
{
    for (int i = 0; i < size; i++) {
      arr[i] = i;
    }
}

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    int size = 1;
    int* arr = new int[size];
    long_operation(arr, size);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
    return 0;
}