import os
import yaml
import re
from typing import Any
from functools import cached_property

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from gplugins.common.base_models.component import LayeredComponentBase
from gplugins.tidy3d.util import get_port_normal, sort_layers

import tidy3d as td

from SimulationSettings import SimulationSettingsTiny3DFdtd, SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

import numpy as np

import tiktoken
import openai
# Configure OpenAI API Key
OPENAI_API_KEY = "ENTER API KEY"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

########################################################################################################################################################

class Tidy3DSimulator():

    def __init__(self, component, settings: SimulationSettingsTiny3DFdtd = SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT):

        self.settings= settings

        #self.comp_base = LayeredComponentBase()
        
        self.component = component

        #self.material_mapping = self.settings.material_mapping
        #self.reference_plane = "middle"
        
        self.simulation = None

        self.comp_base = LayeredComponentBase(component=self.component, layer_stack=self.settings.layer_stack, extend_ports=self.settings.extend_ports,
                                              port_offset=self.settings.port_offset, pad_xy_inner=self.settings.pad_xy_inner, pad_xy_outer=self.settings.pad_xy_outer,
                                              pad_z_inner=self.settings.pad_z, pad_z_outer=self.settings.pad_z)


    @cached_property
    def polyslabs(self) -> dict[str, tuple[td.PolySlab, ...]]:
        """Returns a dictionary of PolySlab instances for each layer in the component.

        Returns:
            dict[str, tuple[td.PolySlab, ...]]: A dictionary mapping layer names to tuples of PolySlab instances.
        """
        slabs = {}
        layers = sort_layers(self.comp_base.geometry_layers, sort_by="mesh_order", reverse=True)

        for name, layer in layers.items():
            if name != "core":
                continue
            bbox = self.comp_base.get_layer_bbox(name)
            slabs[name] = tuple(
                td.PolySlab(
                    vertices=v,
                    axis=2,
                    slab_bounds=(bbox[0][2], bbox[1][2]),
                    sidewall_angle=np.deg2rad(layer.sidewall_angle),
                )
                for v in self.comp_base.get_vertices(name)
            )

        return slabs

    @cached_property
    def structures(self) -> list[td.Structure]:
        """Returns a list of Structure instances for each PolySlab in the component.

        Returns:
            list[td.Structure]: A list of Structure instances.
        """
        structures = []
        
        for name, polys in self.polyslabs.items():
            structures.extend(
                [
                    td.Structure(
                        geometry=poly,
                        medium=self.settings.material_mapping[
                            self.comp_base.geometry_layers[name].material
                        ],
                        name=f"{name}_{idx}",
                    )
                    for idx, poly in enumerate(polys)
                ]
            )
        return structures

    def get_ports(self, mode_spec: td.ModeSpec) -> list[td.plugins.smatrix.Port]:

        ports = []
        for port in self.comp_base.ports:
        
            if port.port_type != "optical":
                continue

            center = self.comp_base.get_port_center(port)
            center = np.round(center, abs(int(np.log10(1e-6)))) # round to the nearest micron
            
            axis, direction = get_port_normal(port)

            match self.settings.port_size_mult:
                case float():
                    size = np.full(3, self.settings.port_size_mult * port.dwidth)
                case tuple():
                    size = np.full(3, self.settings.port_size_mult[0] * port.dwidth)
                    size[2] = self.settings.port_size_mult[1] * port.dwidth
            size[axis] = 0

            ports.append(
                td.plugins.smatrix.Port(
                    center=tuple(center),
                    size=tuple(size),
                    direction=direction,
                    mode_spec=mode_spec,
                    name=port.name,
                )
            )
        return ports
    
    def get_char_ports(self, mode_spec: td.ModeSpec) -> list[td.plugins.smatrix.Port]:

        "List of the first and last port. First port will be used as input, last port will be used as output for characterization"

        ports = []
        for port in [self.comp_base.ports[0], self.comp_base.ports[-1]]:
        
            if port.port_type != "optical":
                continue

            center = self.comp_base.get_port_center(port)
            center = np.round(center, abs(int(np.log10(1e-6)))) # round to the nearest micron
            
            axis, direction = get_port_normal(port)

            match self.settings.port_size_mult:
                case float():
                    size = np.full(3, self.settings.port_size_mult * port.dwidth)
                case tuple():
                    size = np.full(3, self.settings.port_size_mult[0] * port.dwidth)
                    size[2] = self.settings.port_size_mult[1] * port.dwidth
            size[axis] = 0

            ports.append(
                td.plugins.smatrix.Port(
                    center=tuple(center),
                    size=tuple(size),
                    direction=direction,
                    mode_spec=mode_spec,
                    name=port.name,
                )
            )
        return ports
    
    def create_simulation(self,
                          sources: tuple[Any, ...] | None = None,
                          monitors: tuple[Any, ...] | None = None
                          ) -> td.Simulation:

        sim_center = (*self.comp_base.center[:2], self.settings.layer_stack['core'].thickness/2) 
        sim_size = (*self.comp_base.size[:2], self.settings.layer_stack['core'].thickness/2 + 2*self.settings.pad_z)  

        grid_spec = self.settings.grid_spec

        boundary_spec = self.settings.boundary_spec

        return td.Simulation(
            center=sim_center,
            size=sim_size,
            grid_spec=grid_spec,
            medium = self.settings.material_mapping["sio2"],

            structures=self.structures,
            sources=[] if sources is None else sources,
            monitors=[] if monitors is None else monitors,
            boundary_spec=boundary_spec,

            run_time=self.settings.run_time
        )

    def create_modesimulation(self):

        ldas = np.linspace(self.settings.wavelength - self.settings.bandwidth/2, self.settings.wavelength + self.settings.bandwidth/2, 101)  # wavelength range
        
        freq0 = td.C_0/self.settings.wavelength
        freqs = td.C_0 / ldas  # frequency range
        fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
            

        char_ports = self.get_char_ports(td.ModeSpec(num_modes=self.settings.num_modes))

        source_port = char_ports[0]
        monitor_port = char_ports[1]

        mode_source = td.ModeSource(
                        center=source_port.center,
                        size=source_port.size,
                        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
                        direction="+",
                        mode_spec=source_port.mode_spec,
                        mode_index=0,
                        )   

        # add a mode monitor to measure transmission at the output waveguide
        mode_monitor = td.ModeMonitor(
                        center=monitor_port.center,
                        size=monitor_port.size,
                        freqs=freqs,
                        mode_spec=monitor_port.mode_spec,
                        name="mode",
                        )

        # add a field monitor to visualize field distribution at z=t/2
        field_monitor = td.FieldMonitor(
            center=(0, 0, self.settings.layer_stack['core'].thickness/2), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
        )

        mode_sim = self.create_simulation(sources=[mode_source], monitors=[mode_monitor, field_monitor])

        return mode_sim

    def create_fdtdsimulation(self):
        
        ldas = np.linspace(self.settings.wavelength - self.settings.bandwidth/2, self.settings.wavelength + self.settings.bandwidth/2, 101)  # wavelength range
        
        freq0 = td.C_0/self.settings.wavelength
        freqs = td.C_0 / ldas  # frequency range
        fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
        
        field_monitor = td.FieldMonitor(
            center=(0, 0, self.settings.layer_stack['core'].thickness/2), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
        )
        
        fdtd_sim = self.create_simulation(monitors=[field_monitor])
        
        return fdtd_sim

########################################################################################################################################################

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_prompt(prompt, max_tokens=120000):
    # Tokenize the input prompt
    tokens = tokenizer.encode(prompt)
    
    # Check if the prompt exceeds the maximum allowed tokens
    if len(tokens) > max_tokens:
        # Truncate the prompt by keeping only the last `max_tokens` tokens
        tokens = tokens[-max_tokens:]
        
        # Decode tokens back to string
        truncated_prompt = tokenizer.decode(tokens)
        return truncated_prompt
    return prompt

def call_openai(prompt, sys_prompt='', model='gpt-4o'):
    '''
    calling openai. This is my persoanl account.
    '''
    prompt = truncate_prompt(prompt)
    
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        # model="gpt-4",
        # model="gpt-4-turbo",
        model=model,
        temperature= 0.1,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content

def run_llm(simulator: Tidy3DSimulator, log_filepath):
    LOG_MAIN = log_filepath 
    with open(LOG_MAIN, "r", encoding="utf-8") as file:
        log_data = file.read()
    
    SYS_PROMPT = os.getcwd() + "\\SYS_PROMPT.txt" 
    with open(SYS_PROMPT, "r", encoding="utf-8") as file:
        sys_prompt = file.read()

    #print("################################################################################")
    #print("System Prompt Sent to OPENAI")
    #print("################################################################################")
    #print(sys_prompt)
    #print("################################################################################\n\n")

    data_to_send = "\n CURRENT VALUES \n min_steps_per_wvl: %f \n extend_ports: %f \n port_offset: %f \n pad_xy_inner: %f \n pad_xy_outer: %f \n pad_z: %f \n run_time: %s \n shutoff: %s \n wavelength: %f" % (simulator.settings.min_steps_per_wvl, simulator.settings.extend_ports, simulator.settings.port_offset, simulator.settings.pad_xy_inner, simulator.settings.pad_xy_outer, simulator.settings.pad_z, str(simulator.settings.run_time), str(simulator.settings.shutoff), simulator.settings.wavelength )
    
    data_to_send = log_data + data_to_send
    #print("################################################################################")
    #print("User Prompt Sent to OPENAI")
    #print("################################################################################")
    #print(data_to_send)
    #print("################################################################################\n\n")
    
    openai_response = call_openai(data_to_send, sys_prompt)
    
    print("################################################################################")
    print("Response from OPENAI")
    print("################################################################################")
    print(openai_response)
    print("################################################################################\n\n")

    suggested_param = openai_response[7:(len(openai_response)-3)]

    #print(ans)

    suggested_param_yaml = yaml.safe_load(suggested_param)

    #print(yaml_content)

    return suggested_param_yaml

########################################################################################################################################################

def run_modesimulation(simulator: Tidy3DSimulator):

    ldas = np.linspace(simulator.settings.wavelength - simulator.settings.bandwidth/2, simulator.settings.wavelength + simulator.settings.bandwidth/2, 101)  # wavelength range
    
    freqs = td.C_0 / ldas  # frequency range

    modesim = simulator.create_modesimulation()

    modesim.plot(z=0)
    plt.show(block=True)

    mode_spec = td.ModeSpec(num_modes=simulator.settings.num_modes)

    char_ports = simulator.get_char_ports(mode_spec=mode_spec)
    source_port = char_ports[0]

    mode_solver = td.plugins.mode.ModeSolver(
        simulation=modesim,
        plane=td.Box(center=source_port.center, size=source_port.size),
        mode_spec=mode_spec,
        freqs=freqs
    )

    mode_data = td.web.run(mode_solver, task_name="modesim", verbose=True)

    return mode_solver, mode_data

def run_modesimulation_llm(simulator: Tidy3DSimulator):

    try:
        mode_solver, mode_data = run_modesimulation(simulator)

        LOG_MAIN = os.getcwd() + "\\LOG_MAIN.txt"

        warning_found = False

        with open(LOG_MAIN, "r") as file:
            for line in file:
                if "WARNING" in line:
                    warning_found = True


        if warning_found == True:

            suggested_param_yaml = run_llm(simulator, LOG_MAIN)

            sim_settings_new = simulator.settings

            for key, val in suggested_param_yaml.items():
                if hasattr(sim_settings_new, key):
                    setattr(sim_settings_new, key, val)

            tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
                        
            mode_solver, mode_data = run_modesimulation(simulator=tinycomp_new)  

    except:
        print("\n\n################################################################################\n SIM FAILED, ASKING OPENAI \n################################################################################\n\n")

        suggested_param_yaml = run_llm(simulator, os.getcwd() + "\\LOG_MAIN.txt")

        #print(yaml_content)

        sim_settings_new = simulator.settings

        for key, val in suggested_param_yaml.items():
            if hasattr(sim_settings_new, key):
                setattr(sim_settings_new, key, val)

        #sim_settings = load_simulation_settings("SimulationSettingsGPT.py", "SimulationSettingsTiny3DFdtd")

        tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
        
        #tinycomp_new.settings = sim_settings_new
        
        mode_solver, mode_data = run_modesimulation(simulator=tinycomp_new)

    return mode_solver, mode_data

########################################################################################################################################################

def extract_dominant_neff(simulator: Tidy3DSimulator, mode_solver: td.plugins.mode.ModeSolver, mode_output: td.plugins.mode.ModeSolverData):

    dominant_neff_index = -1
    dominant_neff = 0

    neff_data = mode_output.n_eff.sel(f = td.C_0/simulator.settings.wavelength)
    for i in range(len(neff_data)):
        if neff_data[i] > dominant_neff:
            dominant_neff = neff_data[i]
            dominant_neff_index = i

    return dominant_neff

########################################################################################################################################################

def run_fdtdsimulation(simulator: Tidy3DSimulator, dominant_neff: float):

    ldas = np.linspace(simulator.settings.wavelength - simulator.settings.bandwidth/2, simulator.settings.wavelength + simulator.settings.bandwidth/2, 101) 
    
    freqs = td.C_0 / ldas 

    fdtdsim = simulator.create_fdtdsimulation()

    ports = simulator.get_ports(td.ModeSpec(num_modes=1, target_neff=dominant_neff))

    fdtd_solver = td.plugins.smatrix.ComponentModeler(simulation=fdtdsim, ports=ports, freqs=freqs, run_only=simulator.settings.run_only, verbose=True, path_dir="data")

    smatrix = fdtd_solver.run()

    return fdtd_solver, smatrix

def run_fdtdsimulation_llm(simulator: Tidy3DSimulator, dominant_neff: float):
    try:
        fdtd_solver, smatrix = run_fdtdsimulation(simulator, dominant_neff)

        LOG_MAIN = os.getcwd() + "\\LOG_MAIN.txt"
        
        warning_found = False

        with open(LOG_MAIN, "r") as file:
            for line in file:
                if "WARNING" in line:
                    warning_found = True


        if warning_found == True:
            suggested_param_yaml = run_llm(simulator, LOG_MAIN)

            sim_settings_new = simulator.settings

            for key, val in suggested_param_yaml.items():
                if hasattr(sim_settings_new, key):
                    setattr(sim_settings_new, key, val)

            tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
                        
            fdtd_solver, smatrix = run_fdtdsimulation(simulator=tinycomp_new, dominant_neff=dominant_neff)  
            
    except:
        print("\n\n################################################################################\n SIM FAILED, ASKING OPENAI \n################################################################################\n\n")

        suggested_param_yaml = run_llm(simulator, os.getcwd() + "\\LOG_MAIN.txt")

        sim_settings_new = simulator.settings

        for key, val in suggested_param_yaml.items():
            if hasattr(sim_settings_new, key):
                setattr(sim_settings_new, key, val)

        #sim_settings = load_simulation_settings("SimulationSettingsGPT.py", "SimulationSettingsTiny3DFdtd")

        tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
        
        #tinycomp_new.settings = sim_settings_new
        
        fdtd_solver, smatrix = run_fdtdsimulation_llm(simulator=tinycomp_new, dominant_neff=dominant_neff)

    return fdtd_solver, smatrix

########################################################################################################################################################

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

    ### Test Case 3

    # This warning is generated when there is a discontinuous change in the mode profile when frequency increments.
    
    """ if test_case == "Test 3": 
        custom_settings = SimulationSettingsTiny3DFdtd()

        tinycomp = Tidy3DSimulator(component=c, settings=custom_settings)

        fdtd_solver, smatrix = run_fdtdsimulation_llm(tinycomp, dominant_neff=1) """
