# In the name of God

# import libraries:
import pandas as pd
from netmiko import ConnectHandler

### option1: read ips from a text file:
ip_list1 = []

with open('/home/farhad/Desktop/AI-and-Python-Anisa/Session4/ip_list.txt', 'r') as file:
    for line in file:
        ip_list1.append(line.strip())
        print(ip_list1)
### option2: read ips from csv file:
ip_list2 = []
df = pd.read_csv('/home/farhad/Desktop/AI-and-Python-Anisa/Session4/ip_list.csv')
for i in range(len(df)):
    ip_list2.append(df.iloc[i, 1])
    print(ip_list2)

# create a loop for ssh connection:
for ip in ip_list2:
    devices = {
        'device_type': 'cisco_ios',
        'host': ip,
        'username': 'admin',
        'password': '1234',
        'port': 22,
        'session_log': 'log.txt',
        'auth_timeout': 60,
        }

    # ssh to deivece:
    with ConnectHandler(**devices) as ssh:
        print(f'connecting to {ssh.find_prompt()}')
        output = ssh.config_mode()
        output += ssh.send_config_from_file('/home/farhad/Desktop/AI-and-Python-Anisa/Session4/conf_file.txt')
        write_output = ssh.save_config()
    print(output)
    print(write_output)























