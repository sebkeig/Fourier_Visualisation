from presentation import * 
import numpy as np

time_table, x_table, y_table = create_close_loop('katie_seb_louvre_process.jpg')

plt.show()
print(len(x_table))
# order is the number of harmonics
timestep = 1/tau
coordinate_as_complex_V = coordinate_to_complex_number(timestep, time_table, x_table, y_table)

print(coordinate_as_complex_V)

transformed = DFT(timestep, coordinate_as_complex_V)

print(transformed)

