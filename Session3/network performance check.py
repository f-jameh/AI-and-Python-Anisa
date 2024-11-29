# In the name of God

# import libraties
import subprocess as sbp
import matplotlib.pyplot as plt
import sys
import time

# ip = '10.10.10.10'
time_list = []
count_list =[]
monitore_time = 10

# ip = str(sys.argv[1])
ip = '4.2.2.4'

while monitore_time > 0:

    command = sbp.run(f'ping -c 1 -W 2 {ip}', text=True, capture_output=True, shell=True)
    output = command.stdout
    
    if 'time=' in output:

        output = output.split("time=")
        output = float(output[1][0:5])
        monitore_time = monitore_time -1
        time_list.append(output)
        count_list.append(monitore_time)
        print(time_list)
        print(count_list)
        time.sleep(1)
    else:
        print('timeout')

plt.plot(count_list, time_list, color='Blue', marker='o', label='RTT')
plt.xlabel('try')
plt.ylabel('time')
plt.title('network quality graph')
plt.legend()
plt.savefig('/home/farhad/Desktop/plt.png', dpi=300)
plt.show()
plt.close()



