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

// cellular automata simulation is memory bandwidth limited (especially for such simple rule) not memory amount limited when it comes to cell updates per second!

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
