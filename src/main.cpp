/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
 *
 * Using ISO C++ 17 (C++ 11 may be compatible)
 *
 * Conventions (a variation of STL/Boost Style Guides):
 *  - use spaces instead of tabs
 *  - indent with 4 spaces
 *  - variables are camelCased
 *    - params are prefixed with p (e.g. pFillProb)
 *    - member variables are prefixed with m (e.g. mFillProb)
 *    - globals are prefixed with g (e.g. gDisplay)
 *       - the 'config' namespace doesn't follow this as the 'config::' prefix
 *         is always made explicit
 *  - methods are snake_cased
 *  - CUDA kernels are prefixed with k (e.g. k_evolve())
 *  - Macros are UPPER_CASED (e.g. CUDA_ASSERT())
 */
#include <chrono>
#include <iostream>
#include <thread>
#include <signal.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#include "automata_interface.hpp"
#include "automata_base_cpu.hpp"
#include "config.hpp"
#include "pattern.hpp"
#include "stats.hpp"
#include "utils.hpp"

#ifndef CPU_ONLY
#include "automata_bit_gpu.cuh"
#include "automata_base_gpu.cuh"
#endif // CPU_ONLY

#ifdef AUTOMATA_TYPE_BIT
#define AUTOMATA AutomataBit
#else // !AUTOMATA_TYPE_BIT
#define AUTOMATA AutomataBase
#endif // AUTOMATA_TYPE_BIT

// no display stuff should be imported
#ifndef HEADLESS_ONLY
#include "controls.hpp"
#include "display.hpp"
#endif // HEADLESS_ONLY

#ifndef HEADLESS_ONLY
Display *gDisplay;
#endif // HEADLESS_ONLY
AutomataInterface *gAutomata;

bool gLooping = true;
ulong gLastIterationCount = 0;
ulong gIterationsPerSecond = 0;
ulong gNsBetweenSeconds = 0;
std::chrono::steady_clock::time_point gLastPrintClock =
    std::chrono::steady_clock::now();
std::ostringstream gLiveLogBuffer;

void loop();
bool should_log();
void live_log();
void sigint_handler(int s);

int main(int argc, char **argv) {
    spdlog::cfg::load_env_levels();

    const uint randSeed = time(nullptr);
    struct sigaction sigIntHandler;

    // configure interrupt signal handler
    sigIntHandler.sa_handler = sigint_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, nullptr);

    // load command line arguments
    config::load_cmd(argc, argv);

#ifndef HEADLESS_ONLY
    controls::paused = config::startPaused;

    // configure display
    if (config::render)
        gDisplay = new Display(&argc, argv, loop, config::cpuOnly);
#endif // HEADLESS_ONLY

    // configure automata object
    if (config::cpuOnly)
        // the CPU implementation uses the buffer update function provided by
        // the display class and we configure it here to reduce complexity by
        // maintaining the AutomataInterface predictable
        gAutomata = static_cast<AutomataInterface *>(
            new cpu::AutomataBase(randSeed, &gLiveLogBuffer, []() {
#ifndef HEADLESS_ONLY
                gDisplay->update_grid_buffers_cpu();
#endif // HEADLESS_ONLY
            }));
#ifndef CPU_ONLY
#ifndef HEADLESS_ONLY
    else if (config::render)
        // the GPU implementation updates the VBO using the CUDA<>GL interop
        gAutomata = dynamic_cast<AutomataInterface *>(new gpu::AUTOMATA(
            randSeed, &gLiveLogBuffer, &(gDisplay->grid_vbo())));
#endif // HEADLESS_ONLY
    else
        gAutomata = dynamic_cast<AutomataInterface *>(
            new gpu::AUTOMATA(randSeed, &gLiveLogBuffer));
#endif // CPU_ONLY

    if (config::patternFileName != "random")
        load_pattern(config::patternFileName);

    // prepare to start loop
    gAutomata->prepare();

    spdlog::info("Running evolution loop...");

#ifndef HEADLESS_ONLY
    if (config::render)
        gDisplay->start();
    else
#endif // HEADLESS_ONLY
        while (gLooping)
            loop();

    if (config::benchmarkMode)
        stats::print_timings();
    else if (config::printOutput)
        utils::print_output();
    else
        std::cout << std::endl;

    spdlog::info("Exiting after {} iterations.", stats::iterations);

    // clean up
    delete gAutomata;

#ifndef HEADLESS_ONLY
    if (config::render)
        delete gDisplay;
#endif // HEADLESS_ONLY

    return 0;
}

void loop() {
    // limit framerate
    if (config::renderDelayMs > 0)
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config::renderDelayMs));

    // loop timer
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    // prepare logging
    const bool logEnabled = !config::benchmarkMode && should_log();
    if (logEnabled)
        // carriage return
        gLiveLogBuffer << "\r\e[KIt: " << stats::iterations;

#ifndef HEADLESS_ONLY
    // update buffers & render
    if (config::render) {
        // update display buffers
        gAutomata->update_grid_buffers();

        // display current grid
        gDisplay->draw(logEnabled, gIterationsPerSecond);
    }

    // there are controls only when rendering is enabled
    if (controls::paused && !controls::singleStep) {
        if (config::render)
            std::cout << "\r\e[KPaused. Press space to resume." << std::flush;
    } else {
        // compute a batch of generations if not rendering a single step
        if (!controls::singleStep)
            for (uint repeat = 0; repeat < config::skipFrames; ++repeat) {
                // compute extra generations
                gAutomata->evolve(false);
                ++stats::iterations;
            }

        controls::singleStep = false;

#endif // HEADLESS_ONLY

        // compute next grid
        gAutomata->evolve(logEnabled); // count alive cells if will log
        ++stats::iterations;

#ifndef HEADLESS_ONLY
    }
#endif // HEADLESS_ONLY

    // calculate loop time and iterations per second
    gNsBetweenSeconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - timeStart)
                             .count();
    if (logEnabled)
        live_log();

    // check if number of iterations reached max
    if (!gLooping || (config::maxIterations > 0 &&
                      stats::iterations >= config::maxIterations)) {
#ifndef HEADLESS_ONLY
        if (config::render)
            gDisplay->stop();
        else
#endif // HEADLESS_ONLY
            gLooping = false;
    }
}

bool should_log() {
    // calculate loop time and iterations per second
    gIterationsPerSecond = stats::iterations - gLastIterationCount;
    // return if it's not time to update the log
    if (gIterationsPerSecond <= 0 ||
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - gLastPrintClock)
                .count() < 1)
        return false;
    return true;
}

void live_log() {
    auto cells = config::rows * config::cols;
    // add main loop info to the buffer

auto n = 2 * cells * gIterationsPerSecond * sizeof(GridType) ;
auto t = 448000000000;
auto p = (n * 100.) / t;
    // The RTX 2080 has 2944 CUDA cores and a boost clock at 1800MHz. 8GB of GDDR6 memory features a 256-bit bus for 448GB/sec of 
    // In the end, the GeForce RTX 2080 has 46 SMs that contain 2944 CUDA cores, 368 tensor cores, 46 RT cores, 184 texture units, 64 ROPS, and 4 MB of L2 cache.
    gLiveLogBuffer << " | It/s: " << gIterationsPerSecond
        << " | main memory rw 2xcells|bytes/s: " << n << "= " << p << "% of RTX 2080 theoretical 448000000000 bytes/s"
                   << " | Main Loop: "
                   // average time per iteration
                   << gNsBetweenSeconds / gIterationsPerSecond << " ns";

// 30k * 30k (ca 1 GB of memory for the grid)
// It: 1072 | Evolve Kernel: 21568882 ns | Active cells: 38123233 | It/s: 51 | main memory rw 2xcells|bytes/s: 91800000000= 20.4911% of RTX 2080 theoretical 448000000000 bytes/s | Main Loop: 20200426 ns

    // TODO for another 5x, improve the memory access pattern... maybe explicitly use shared memory for the few reused cells, striding rightwards and upwards in the grid, find ideal parameters...

// note: cpu memory bandwith is about 20x lower:
// https://codearcana.com/posts/2013/05/18/achieving-maximum-memory-bandwidth.html
// https://github.com/awreece/memory-bandwidth-demo
/*
ubuntu@ubuntu44:~/memory-bandwidth-demo$ ./memory_profiler 
           read_memory_rep_lodsq: 10.07 GiB/s
                read_memory_loop: 12.54 GiB/s
                 read_memory_sse: 11.98 GiB/s
                 read_memory_avx: 12.50 GiB/s
        read_memory_prefetch_avx:  7.95 GiB/s
               write_memory_loop: 12.44 GiB/s
          write_memory_rep_stosq: 22.97 GiB/s
                write_memory_sse: 12.87 GiB/s
    write_memory_nontemporal_sse: 22.17 GiB/s
                write_memory_avx: 12.46 GiB/s
    write_memory_nontemporal_avx: 22.12 GiB/s
             write_memory_memset: 22.97 GiB/s
memory_profiler: main.c:51: timeitp: Assertion `SIZE % omp_get_max_threads() == 0' failed.
Aborted (core dumped)

*/
// but probably the cpu implementation sucks...
/*
It: 44 | Evolve Function: 342304009 ns | Active cells: 104066846 | It/s: 4 | main memory rw 2xcells|bytes/s: 7200000000= 1.60714% of RTX 2080 theoretical 448000000000 bytes/s | Main Loop: 412621445 ns^C

7 200 000 000 <- well 7 GB s

that is somewhere in the ballpark...

wow this system has bad RAM access...
*/

// cellular automata simulation is memory bandwidth limited (especially for such simple rule) not memory amount limited when it comes to cell updates per second!

// https://github.com/bryanoliveira/cellular-automata/issues/10
/*
 Please clarify: This program is memory bandwidth limited, nothing else #10
Open
Masterxilo opened this issue Nov 23, 2023 · 2 comments
Open
Please clarify: This program is memory bandwidth limited, nothing else
#10
Masterxilo opened this issue Nov 23, 2023 · 2 comments
Comments
@Masterxilo
Masterxilo commented Nov 23, 2023

I think it would help the next person learning about this stuff to mention that in your readme.

That's why you don't see much speedup when adding more cores.

I measured/calculated 20% memory bandwidth efficiency for the GPU implementation and 25-50% for the CPU implementation.

None of your evaluations or observations mention memory bandwidth at all.

Also, using curand is what makes your implementation not support even a 30k * 30k = 900 MB grid. sizeof(curandState) = 48 bytes...
If you get rid of curand and use any other decent hash, you can max out the size of the grid.
@Masterxilo
Author
Masterxilo commented Nov 23, 2023 •

your own calculation/measurement also shows that the program is not quite reaching the memory bandwidth bound:

13500*13500*729*2 = 265720500000, 266 GB/s, which is a bit short of the ca 760 GB/s of these cards (rtx 3080)
@Masterxilo
Author
Masterxilo commented Nov 23, 2023

It would be interesting to study how many instructions could be performed during the time of waiting for the memory.

Probably, the amount of cell updates per second could be scaled far beyond the memory bandwidth by doing multiple update steps without accessing main memory in between, but of course then you have to be more careful with coordinating the memory access or overlapping the update regions (and thus performing some redundant updates) to ensure all interactions propagate correctly (information flows at a rate of one cell per update, and to avoid race conditions/nondeterministic updates, you should probably not access memory updated by other thread blocks...).

*/

    // print the buffer
    std::cout << gLiveLogBuffer.str() << std::flush;
    // reset the buffer
    gLiveLogBuffer.str("");
    gLiveLogBuffer.clear();
    // update global counters
    gNsBetweenSeconds = 0;
    gLastIterationCount = stats::iterations;
    gLastPrintClock = std::chrono::steady_clock::now();
}

void sigint_handler(int s) {
    gLooping = false;
    std::cout << std::endl;
}
