
import os

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

import tidy3d as td

from llm_simulator import Tidy3DSimulator, run_fdtdsimulation_llm
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

    # This warning is generated when the run_time or shutoff parameter in the SimulationSetting File is too small.
    
    custom_settings = SimulationSettingsTiny3DFdtd(run_time=1e-13)

    tinycomp = Tidy3DSimulator(component=c, settings=custom_settings)

    fdtd_solver, smatrix = run_fdtdsimulation_llm(tinycomp, dominant_neff=1)