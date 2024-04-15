import numpy as np
t_end = 4.
p0 = 0.
pump_max_pressure = 3.5e4
# pump_max_pressure = 1.e7
# pump_vacuum_pressure = -1.e3
pump_vacuum_pressure = 0.
stroke_index = 0
actuation_frequency = 0.5 # Hz
# actuation_frequency = 0.125 # Hz
stroke_period = 1./actuation_frequency/2    # 2 strokes per cycle
num_strokes = int(actuation_frequency*t_end)*2+1   # 2 strokes per cycle
# time_constant = 8*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
time_constant = 6*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
# time_constant = 3*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.

def compute_chamber_pressure_function(t, pressure_inputs, time_constant, p0, evaluation_t):
    if len(t) != len(pressure_inputs):
        raise ValueError('t and pressure_inputs must have the same length')
    
    # total_t = np.zeros((len(t),len(evaluation_t)))
    chamber_pressure = np.zeros((len(evaluation_t),))
    index = 0
    for i in range(len(t)):   # For each stroke
        # if i < len(t)-1:
        #     evaluation_t = np.linspace(t[i], t[i+1], num_evaluation_points)
        # else:
        #     evaluation_t = np.linspace(t[i], t_end, num_evaluation_points)
        # total_t[i] = evaluation_t
        j = 0
        if i < len(t)-1:
            while evaluation_t[index] < t[i+1]:
                if i == 0:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index], p0, pressure_inputs[i], time_constant)
                else:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index] - t[i],
                                                                        chamber_pressure[index-j-1],
                                                                        pressure_inputs[i],
                                                                        time_constant)
                j += 1
                index += 1
        else:
            while index < len(evaluation_t):
                if i == 0:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index], p0, pressure_inputs[i], time_constant)
                else:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index] - t[i],
                                                                        chamber_pressure[index-j-1],
                                                                        pressure_inputs[i],
                                                                        time_constant)
                index += 1
                j += 1
        i += 1
            
    return chamber_pressure



def stroke_pressure(t, initial_pressure, final_pressure, time_constant):
    return final_pressure \
        - (final_pressure - initial_pressure)*np.exp(-time_constant*t)\
        # + initial_pressure
    # final pressure is what it should go to. There is a decaying exponential term that goes to zero. Initial pressure is to ensure continuity.
    # t is the time WITHIN THE STROKE. This means that the time previous to the stroke is subtracted off before being passed into this function.


left_chamber_inputs = []
right_chamber_inputs = []
for stroke_index in range(num_strokes):
    if stroke_index % 2 == 0:
        right_chamber_inputs.append(pump_max_pressure)
        left_chamber_inputs.append(pump_vacuum_pressure)
    else:
        right_chamber_inputs.append(pump_vacuum_pressure)
        left_chamber_inputs.append(pump_max_pressure)

t = np.linspace(0, t_end, int(num_strokes))
t[1:] = t[1:] - stroke_period/2
evaluation_t = np.linspace(0, t_end, 401)
left_chamber_pressure_values = compute_chamber_pressure_function(t, left_chamber_inputs, time_constant, p0, evaluation_t)
right_chamber_pressure_values = compute_chamber_pressure_function(t, right_chamber_inputs, time_constant, p0, evaluation_t)

# def left_chamber_pressure(t):
#     if t >= stroke_index*t_end + stroke_period/2:
#         stroke_index += 1

#     if stroke_index%2 == 0:
#         left_chamber_pressure_input = 'HIGH'
#     else:
#         left_chamber_pressure_input = 'LOW'

#     if left_chamber_pressure_input == 'HIGH':
#         left_chamber_final_pressure = pump_max_pressure
#     elif left_chamber_pressure_input == 'LOW':
#         left_chamber_final_pressure = pump_vacuum_pressure
#     elif left_chamber_pressure_input == 'OFF':
#         left_chamber_final_pressure = 0
#     else:
#         raise ValueError('Invalid left chamber pressure input: {}'.format(left_chamber_pressure_input))
    
#     if stroke_index == 0:     # first half stroke with p0 as initial pressure
#         left_chamber_pressure_value = stroke_pressure(t,
#                                                       p0,
#                                                       left_chamber_final_pressure,
#                                                       time_constant,
#                                                       )
#     else:
#         left_chamber_pressure_value = stroke_pressure(
#                                                     t,
#                                                     stroke_pressure(t=(stroke_index-1)*t_end+stroke_period/2),
#                                                     left_chamber_final_pressure,
#                                                     time_constant,
#                                                     )

#     return -left_chamber_pressure_value # negative because compressive (positive) pressures correspond to negative tractions.
    


# def left_chamber_pressure(t):
#     if t < t_end/4:
#         return -(pump_max_pressure - pump_max_pressure*np.exp(-2*t/(t_end/3)))
#     elif t < 2*t_end/3:
#         return -(pump_vacuum_pressure + (pump_max_pressure-pump_vacuum_pressure)*np.exp(-2*(t-t_end/3)/(t_end/3)) - pump_max_pressure*np.exp(-2*(t_end/3)/(t_end/3)))
#     else:
#         return -(pump_vacuum_pressure + (pump_max_pressure-pump_vacuum_pressure)*np.exp(-2*((2*t_end/3)-t_end/3)/(t_end/3)) \
#                  - pump_max_pressure*np.exp(-2*(t_end/3)/(t_end/3)) \
#                     + pump_max_pressure - pump_vacuum_pressure - (pump_max_pressure-pump_vacuum_pressure)*np.exp(-2*(t-2*t_end/3)/(t_end/3)))

# def right_chamber_pressure(t):
#     if t < t_end/3:
#         return -(pump_vacuum_pressure - pump_vacuum_pressure*np.exp(-2*t/(t_end/3)))
#     elif t < 2*t_end/3:
#         return -(pump_max_pressure - (pump_max_pressure-pump_vacuum_pressure)*np.exp(-2*(t-t_end/3)/(t_end/3)) - pump_vacuum_pressure*np.exp(-2*(t_end/3)/(t_end/3)))
#     else:
#         return -(pump_vacuum_pressure  + (pump_max_pressure-pump_vacuum_pressure)*np.exp(-2*((2*t_end/3)-t_end/3)/(t_end/3)) + pump_vacuum_pressure*np.exp(-2*(t_end/3)/(t_end/3)))# - (pump_max_pressure-pump_vacuum_pressure)*np.exp(-2*(t-2*t_end/3)/(t_end/3)))
    
import matplotlib.pyplot as plt
# t_test = np.linspace(0, t_end, 100)
# left_chamber_pressure_vector = np.vectorize(left_chamber_pressure)
# right_chamber_pressure_vector = np.vectorize(right_chamber_pressure)
# plt.plot(t_test, -left_chamber_pressure_vector(t_test))
# plt.plot(t_test, -right_chamber_pressure_vector(t_test))
# plt.show()

plt.plot(evaluation_t, left_chamber_pressure_values/1.e3, label='Left chamber')
plt.plot(evaluation_t, right_chamber_pressure_values/1.e3, label='Right chamber')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.title('Chamber pressures')
plt.show()