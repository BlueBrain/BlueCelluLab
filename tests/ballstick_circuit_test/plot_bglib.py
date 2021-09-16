import bluepy

s = bluepy.Simulation("BlueConfig")

# excitatory cell with an incoming inhibitory synapse
v_a1 = s.reports.soma.time_series(1)
# inhibitory cell with an incoming excitatory synapse
v_a2 = s.reports.soma.time_series(2)

t = s.reports.soma.time_range

plot(t, v_a1, 'r-')
plot(t, v_a2, 'b-')
