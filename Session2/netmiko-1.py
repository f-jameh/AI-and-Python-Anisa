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

command = 'show run'

for device in (ip_lists):
    with ConnectHandler(**device) as ssh:
        backupname = ssh.find_prompt().strip('#') + '_'
        dir = '/home/farhad/Desktop/'
        date = str(datetime.date.today())
        backupname = dir + backupname + date
        output = ssh.send_command(command)
        ssh.save_config()
        print(output)
        with open (backupname, 'w') as file:
            file.write(output)


        time.sleep(2)





