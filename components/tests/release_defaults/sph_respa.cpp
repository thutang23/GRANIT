#include "src/core/sph/sph_respa_schedule.h"
#include "src/domain/distributed_fmm_depth_controller.h"
#include "src/domain/rcb_work_model.h"
#include <sstream>
#include <iostream>
#include <stdexcept>
using namespace cudadem;
static void require(bool ok) { if (!ok) throw std::runtime_error("SPH macro/controller regression"); }
static std::string snapshot(const DistributedFMMDepthController& c) {
    std::ostringstream out(std::ios::binary); c.save_checkpoint(out); return out.str();
}
int main() {
    const auto nominal=plan_sph_macro(0.,.1,2,1.);
    require(nominal.count==2 && nominal.dt==.1 && nominal.end_time==.2);
    for (double event : {.03,.17,.2}) {
        const auto clipped=plan_sph_macro(0.,.1,2,event);
        require(clipped.count==2 && clipped.end_time==event && clipped.dt==event/2);
    }
    const auto duration=plan_sph_macro(0.,.03,1,1.,.1);
    require(duration.count==4 && duration.dt==.025 && duration.end_time==.1);
    require(next_sph_output_time(.3,0.,.1) > .3);
    require(next_sph_output_time(.15,.05,.1)==.25);
    bool rejected=false;
    try { plan_sph_macro(1.,0.,2,2.); } catch (...) { rejected=true; }
    require(rejected);
    AdaptiveDepthConfig config; config.enabled=true; config.min_depth=2; config.max_depth=5;
    config.trial_interval_evaluations=2; config.trial_warmup_evaluations=1;
    config.trial_samples=2; config.confirmation_trials=1;
    DistributedFMMDepthController original;
    require(original.initialize(config,1,{3}));
    // Snapshot every phase, then replay the same subsequent measured epoch.
    // This exercises live A/B trial state, cadence, diagnostics and mailboxes.
    for (uint64_t epoch=1; epoch<=40; ++epoch) {
        const auto bytes=snapshot(original);
        DistributedFMMDepthController resumed;
        std::istringstream in(bytes,std::ios::binary); resumed.restore_checkpoint(in,config);
        require(snapshot(resumed)==bytes);
        const auto command=original.command_for_gpu(0);
        const auto restored=resumed.command_for_gpu(0);
        require(command.launch_depth==restored.launch_depth && command.trial_id==restored.trial_id);
        DistributedFMMDepthEpochTiming t; t.epoch=epoch; t.command_trial_id=command.trial_id;
        t.launch_depth=command.launch_depth; t.elapsed_ns=1000000/(1+command.launch_depth);
        t.local_owned_particles=125;
        require(original.publish_timing(0,t) && resumed.publish_timing(0,t));
        original.process_ready_epochs(); resumed.process_ready_epochs();
        require(snapshot(original)==snapshot(resumed));
    }
    rejected=false;
    try {
        DistributedFMMDepthController broken; std::istringstream in(snapshot(original).substr(0,32));
        broken.restore_checkpoint(in,config);
    } catch (...) { rejected=true; }
    require(rejected);
    RCBWorkCalibrator calibration;
    const RCBWorkKey key{}; calibration.set_enabled_keys({key});
    RCBActiveSample sample; sample.epoch=1; sample.active_seconds=.02; sample.work_units=100;
    calibration.observe_global_epoch(1,{sample});
    std::ostringstream calibration_bytes(std::ios::binary); calibration.save_checkpoint(calibration_bytes);
    RCBWorkCalibrator restored_calibration;
    std::istringstream calibration_input(calibration_bytes.str(),std::ios::binary);
    restored_calibration.restore_checkpoint(calibration_input);
    require(restored_calibration.ready()==calibration.ready());
    require(restored_calibration.coefficient(key)==calibration.coefficient(key));
    sample.epoch=2; sample.active_seconds=.03;
    calibration.observe_global_epoch(2,{sample}); restored_calibration.observe_global_epoch(2,{sample});
    require(restored_calibration.coefficient(key)==calibration.coefficient(key));
    std::cout << "SPH macro events and controller continuation passed\n";
}
