import csv
from random import randint
import numpy as np

class Datamanager:
    """ Takes a CSV file exported from OpenRocket
        Will parse and return the given data
            Based on the parameters intiliazed will add noise to the data
            Can also get real measurments
    """

    

    def __init__(self, file_name, vert_acc_accuracy=1, lat_acc_accuracy=1, barometer_accuracy=1, start_t=10, std=False) -> None:
        
        self.data = dict() 

    
        # std deviation
        self.vert_acc_accuracy = vert_acc_accuracy
        self.lat_acc_accuracy = lat_acc_accuracy
        self.barometer_accuracy = barometer_accuracy
        # Set later
        self.delta_t = None
        self.start_t = start_t

        self.std = std
        self.cur_t = start_t

        keys = ['altitude', 'vertical_velocity', 'vertical_acceleration', 'total_velocity', 'total_acceleration', 'position_east_of_launch', 'position_north_of_launch', 'lateral_distance', 'lateral_direction', 'lateral_velocity', 'lateral_acceleration', 'latitude', 'longitude', 'gravitational_acceleration', 'angle_of_attack', 'roll_rate', 'pitch_rate', 'yaw_rate', 'mass', 'propellant_mass', 'longitudinal_moment_of_inertia', 'rotational_moment_of_inertia', 'cp_location', 'cg_location', 'stability_margin_calibers', 'mach_number', 'reynolds_number', 'thrust', 'drag_force', 'drag_coefficient', 'axial_drag_coefficient', 'friction_drag_coefficient', 'pressure_drag_coefficient', 'base_drag_coefficient', 'normal_force_coefficient', 'pitch_moment_coefficient', 'yaw_moment_coefficient', 'side_force_coefficient', 'roll_moment_coefficient', 'roll_forcing_coefficient', 'roll_damping_coefficient', 'pitch_damping_coefficient', 'reference_length', 'reference_area', 'vertical_orientation', 'lateral_orientation', 'wind_velocity', 'air_temperature', 'air_pressure', 'speed_of_sound', 'simulation_time_step', 'computation_time']
        
        with open(file_name, encoding="cp1252") as cf:
            reader = csv.reader(cf)
            for r in reader:
                if r[0][0] == "#": continue
                self.data[float(r[0])] = dict(zip(keys,[float(x) for x in r[1:]]))
                if not self.delta_t:
                    self.delta_t = self.data[float(r[0])]['simulation_time_step']



    def _get_value_noise(self, measure_n, key, noise):
        time = measure_n*self.delta_t + self.start_t
        if time not in self.data:
            print(time)
            return None
        else:
            n = self.data[time][key]
            if n != float("nan"):
                vals = [] 
                if not self.std:
                    p = (n/100.0)*noise
                    print(n, p)
                    vals = np.linspace(n-p, n+p, 1000)
                else:
                    vals = np.linspace(n-noise, n+noise, 1000)
                return vals[randint(0, 999)]




dm = Datamanager("raw_data.csv")

print(dm._get_value_noise(1, "altitude", 10))