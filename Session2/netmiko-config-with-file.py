from netmiko import ConnectHandler
import time
import datetime

device1 = {
    "device_type": "cisco_ios",
    "host": '100.100.100.2',
    "username": 'root',
    "password": '1234',
    "port": "22",
}
device2 = {
    "device_type": "cisco_ios",
    "host": '200.200.200.2',
    "username": 'root',
    "password": '1234',
    "port": "22",
}
device3 = {
    "device_type": "cisco_ios",
    "host": '200.200.250.2',
    "username": 'root',
    "password": '1234',
    "port": "22",
}

ip_lists = [device1, device2, device3]

config_file = '/home/farhad/Desktop/R2_2024-11-15'

for device in (ip_lists):
    with ConnectHandler(**device) as ssh:
        output = ssh.send_config_from_file(config_file)
        ssh.save_config()
        print(output)


        time.sleep(2)





