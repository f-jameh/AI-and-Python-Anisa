# In the name of God

# import libraries
import subprocess
import time
from netmiko import ConnectHandler

# define variable
vm_ip = '172.16.100.122'
host1_ip = '172.16.100.171'
host2_ip = '172.16.100.172'
off_command = 'shutdown now'
define_command = 'virsh define --file alp1.xml'
start_command = 'virsh start --domain alp-ub1'

host1_ssh = {
    'device_type': 'linux',
    'host': host1_ip,
    'username': 'root',
    'password': 'g',
    'port': 22}

host2_ssh = {
    'device_type': 'linux',
    'host': host2_ip,
    'username': 'farhad',
    'password': 'g',
    'port': 22}


# chack service health with curl
while True:
    result = subprocess.getoutput(f'curl {vm_ip}')
    if 'works' in result:
        print('site is up')
        time.sleep(3)
    # transfer and run the vm in case of failure    
    else:
        try:
            with ConnectHandler(**host1_ssh) as net_connect:
                output = net_connect.send_command(off_command)     # poweroff vm/host
        except:
            print(f'service {vm_ip} is down now')
            subprocess.run(f'scp /home/farhad/Desktop/AI-and-Python-Anisa/Session9/alp-ub1.xml {host2_ip}:/home/farhad/alp1.xml', shell=True, capture_output=False)
            print('file successfully transfered')
            with ConnectHandler(**host2_ssh) as net_connect:
                output = net_connect.send_command(define_command)     # define vm from xml file in host2
            with ConnectHandler(**host2_ssh) as net_connect:
                output = net_connect.send_command(start_command)     # start created vm again in host2
            print()
            print(output)
            print()
            time.sleep(1)








