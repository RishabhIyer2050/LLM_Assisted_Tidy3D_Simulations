import os

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

import tidy3d as td

from llm_simulator import Tidy3DSimulator, run_modesimulation_llm
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

    # This error is generated when the port_offset parameter in the Simulation Setting File to 0. 

    custom_settings = SimulationSettingsTiny3DFdtd(port_offset=0)

    tinycomp = Tidy3DSimulator(component=c, settings=custom_settings)

    _, mode_data = run_modesimulation_llm(tinycomp)

    print(mode_data.to_dataframe())