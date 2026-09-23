// =============================================================================
// SPH Configuration Printer
// =============================================================================
// Prints SPH-specific configuration parameters not covered by print_config().
// Called from main_sph.cu when SPH is enabled.

#ifndef CUDADEM_SPH_CONFIG_PRINTER_H
#define CUDADEM_SPH_CONFIG_PRINTER_H

#include "src/core/constant/command_line_interface.h"
#include <iostream>

namespace cudadem {

// Print SPH-specific configuration parameters to stdout
// Should be called after print_config() when config.sph.enabled == true
void print_sph_config(const SimConfig& config);

} // namespace cudadem

#endif // CUDADEM_SPH_CONFIG_PRINTER_H
