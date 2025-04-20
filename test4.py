import os
import re

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

import tidy3d as td

from llm_simulator import Tidy3DSimulator, run_fdtdsimulation_llm, run_fdtdsimulation, run_llm
from SimulationSettings import SimulationSettingsTiny3DFdtd, SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT

if __name__ == "__main__":

    ### Test Device

    # A 1x2 MMI is arbitarily choosen for testing. This device is defined using GDSFactory's Component Library. 

    pdk = get_generic_pdk()
    pdk.activate()

    c = gf.Component()
    m = gf.components.mmi1x2(length_mmi=12.8, width_mmi=3.8, gap_mmi=0.25, length_taper=10.0, width_taper=1.4)
    coupler_r = c << m
    c.add_port("o1", port=coupler_r.ports["o1"])
    c.add_port("o2", port=coupler_r.ports["o2"])
    c.add_port("o3", port=coupler_r.ports["o3"])
    c.flatten()

    td.config.logging_level = "DEBUG"

    LOG_MAIN = os.getcwd() + "\\LOG_MAIN.txt" 
    td.set_logging_console(stderr=False)

    td.set_logging_file(fname = LOG_MAIN, level="DEBUG", filemode='w')

    # This warning is generated when there is a discontinuous change in the mode profile when frequency increments.
    
    custom_settings = SimulationSettingsTiny3DFdtd(run_only=(("o1",0), ("o1",0)))

    tinycomp = Tidy3DSimulator(component=c, settings=custom_settings)

    fdtd_solver, smatrix = run_fdtdsimulation(tinycomp, 1)

    with open(LOG_MAIN, "r", encoding="utf-8") as file:
        log_main = file.read()

    task_ids = []
    task_ids = re.findall(r'fdve-[a-f0-9\-]+', log_main)


    LOG_SUB = os.getcwd() + "\\LOG_SUB.txt" 
    for i in task_ids:
        td.web.download_log(i)

        with open("tidy3d.log", "r", encoding="utf-8") as file:
            log_sub = file.read()

        with open(LOG_SUB, "w", encoding="utf-8") as file:
            file.write(log_sub.strip())

        with open(LOG_SUB, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if "WARNING" in line:
                print(line.strip())
                # Print next two lines, if they exist
                if i + 1 < len(lines):
                    print(lines[i + 1].strip())
                if i + 2 < len(lines):
                    print(lines[i + 2].strip())

    update_vars = run_llm(tinycomp, LOG_SUB)

    sim_settings_new = tinycomp.settings

    for key, val in update_vars.items():
        if hasattr(sim_settings_new, key):
            setattr(sim_settings_new, key, val)

    tinycomp_new = Tidy3DSimulator(component=tinycomp.component, settings=sim_settings_new)
                
    _, mode_data = run_fdtdsimulation(simulator=tinycomp_new, dominant_neff=1)
    

