#include <boost/program_options.hpp>
#include <iostream>

#include "config.hpp"

namespace po = boost::program_options;

namespace config {

std::string programName = "Automata";
unsigned int width = 600;
unsigned int height = 600;
// 12000 x 12000 uses up to 2GB RAM and 8.5GB VRAM
unsigned int rows = 1200;
unsigned int cols = 1200;
bool render = false;
unsigned int renderDelayMs = 0;
float fillProb = 0.2;
float virtualFillProb = 0;
unsigned long maxIterations = 0;
bool cpuOnly = false;
std::string patternFileName("random");
bool startPaused = true;
bool noDownsample = false;

void load_file() {}
void load_cmd(int argc, char **argv) {
    po::options_description description("Usage");

    description.add_options()("help,h", "Display this help message") //
        ("width", po::value<unsigned int>(), "Window width")         //
        ("height", po::value<unsigned int>(), "Window height")       //
        ("rows,y", po::value<unsigned int>(), "Grid rows")           //
        ("cols,x", po::value<unsigned int>(), "Grid cols")           //
        ("render,r",
         "Enable render (default is to run in headless mode)") //
        ("render-delay,d", po::value<unsigned int>(),
         "Render delay between frames (in milliseconds)") //
        ("fill-probability,p", po::value<float>(),
         "Cell probability to start alive") //
        ("virtual-fill-probability,v", po::value<float>(),
         "Cell probability to become alive")                    //
        ("max,m", po::value<unsigned long>(), "Max iterations") //
        ("cpu", "Enable CPU-only mode")                         //
        ("no-downsample", "Disable automatic grid to vertice downsampling"
                          " when grid size is greater than window size.") //
        ("file,f", po::value<std::string>(), "Pattern file (.rle)")       //
        ("start", "Unpause at start (default is paused when rendering, "
                  "unpaused when not rendering)"); //

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << description << std::endl;
        exit(0);
    }
    if (vm.count("width"))
        width = vm["width"].as<unsigned int>();
    if (vm.count("height"))
        height = vm["height"].as<unsigned int>();
    if (vm.count("rows"))
        rows = vm["rows"].as<unsigned int>();
    if (vm.count("cols"))
        cols = vm["cols"].as<unsigned int>();
    if (vm.count("render"))
        render = true;
    if (vm.count("render-delay"))
        renderDelayMs = vm["render-delay"].as<unsigned int>();
    if (vm.count("fill-probability"))
        fillProb = vm["fill-probability"].as<float>();
    if (vm.count("virtual-fill-probability"))
        virtualFillProb = vm["virtual-fill-probability"].as<float>();
    if (vm.count("max"))
        maxIterations = vm["max"].as<unsigned long>();
    if (vm.count("cpu"))
        cpuOnly = true;
    if (vm.count("file"))
        patternFileName = vm["file"].as<std::string>();
    if (vm.count("start") || !render)
        // by default, start the computation loop automatically if we're not
        // rendering
        startPaused = false;
    if (vm.count("no-downsample") || (width == cols && height == rows))
        // by default, don't use downsample when scale is 1:1
        noDownsample = true;
}

} // namespace config
