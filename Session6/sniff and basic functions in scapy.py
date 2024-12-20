# In the name of God

from scapy.all import RandMAC, RandIP, RandIP6, getmacbyip, sniff, wrpcap

random_mac = RandMAC()
print(f'Random MAC is: {random_mac}')

random_mac = RandIP()
print(f'Random IP is: {random_mac}')


random_mac = RandIP6()
print(f'Random IPv6 is: {random_mac}')

mac = getmacbyip('10.10.10.1')
print(f'mac  is: {mac}')

Packets = sniff(iface='wlp1s0', timeout=4, count=10, filter='icmp')

print('######## entire list ####################')
Packets.show()

print('######## only second packet ####################')
Packets[1].show()

print('######## only second packet and ICMP layer ####################')
Packets[1]['ICMP'].show()

print('######## only second packet and src ip ####################')
print(Packets[1]['IP'].src)

print('######## change a field in a packet ####################')
Packets[1]['IP'].src = '10.10.100.100'
Packets[1].show()

file_name = '/home/farhad/Desktop/captured_packet1.pcap'
wrpcap (file_name, Packets)
print(f'all captured packets are stored in {file_name}')




# Filters

# 1. Basic Protocol Filters
# "ip"	                                       Capture all IPv4 packets.         	sniff(filter="ip", count=10)
# "icmp"	                                   Capture ICMP (ping) packets.	        sniff(filter="icmp")
# "tcp"	                                       Capture all TCP packets.	            sniff(filter="tcp")
# "udp"	                                       Capture all UDP packets.	            sniff(filter="udp")
# "arp"	                                       Capture all ARP packets.	            sniff(filter="arp")
# "ip6"	                                       Capture all IPv6 packets.	            sniff(filter="ip6")

# 2. Port-Based Filters
# "tcp port 80"     	                       Capture TCP packets on port 80 (HTTP).	sniff(filter="tcp port 80", count=10)
# "udp port 53"	                               Capture UDP packets on port 53 (DNS).	sniff(filter="udp port 53", count=10)
# "tcp portrange 8000-8080"	                   Capture TCP packets on ports 8000–8080.	sniff(filter="tcp portrange 8000-8080", count=10)
# "not port 22"	                               Exclude traffic on port 22 (SSH).	sniff(filter="not port 22", count=10)

# 3. Host-Based Filters
# "host 192.168.1.1"	                       Capture traffic to/from a specific host.	sniff(filter="host 192.168.1.1", count=10)
# "src host 192.168.1.1"	                   Capture traffic from a specific source host.	sniff(filter="src host 192.168.1.1", count=10)
# "dst host 192.168.1.1"	                   Capture traffic to a specific destination host.	sniff(filter="dst host 192.168.1.1", count=10)

# 4. Network/Subnet Filters
# "net 192.168.1.0/24"                         Capture traffic in a subnet.	sniff(filter="net 192.168.1.0/24", count=10)
# "src net 10.0.0.0/8"	                       Capture traffic from a specific source subnet.	sniff(filter="src net 10.0.0.0/8", count=10)
# "dst net 172.16.0.0/12"                      Capture traffic to a specific destination subnet.	sniff(filter="dst net 172.16.0.0/12", count=10)

# 5. Ethernet Filters
# "ether host 00:11:22:33:44:55"	           Capture traffic to/from a specific MAC address.	sniff(filter="ether host 00:11:22:33:44:55", count=10)
# "ether src 00:11:22:33:44:55"	               Capture traffic from a specific source MAC.	sniff(filter="ether src 00:11:22:33:44:55", count=10)
# "ether dst 00:11:22:33:44:55"                Capture traffic to a specific destination MAC.	sniff(filter="ether dst 00:11:22:33:44:55", count=10)

# 6. Advanced and combination Filters
# "tcp and port 80"	                           Capture TCP packets on port 80.	sniff(filter="tcp and port 80", count=10)
# "udp and port 53 and src host 8.8.8.8"	       Capture DNS replies from Google Public DNS.	sniff(filter="udp and port 53 and src host 8.8.8.8", count=10)
# "tcp or udp"	                               Capture both TCP and UDP packets.	sniff(filter="tcp or udp", count=10)
# "ip and not tcp"                             Capture all IPv4 packets except TCP.	sniff(filter="ip and not tcp", count=10)
# "host 192.168.1.1 and port 22"               Capture SSH traffic to/from a specific host.	sniff(filter="host 192.168.1.1 and port 22", count=10)
# "less 100"	                               Capture packets smaller than 100 bytes.	sniff(filter="less 100", count=10)
# "greater 512"	                               Capture packets larger than 512 bytes.	sniff(filter="greater 512", count=10)
# "vlan 10"	                                   Capture traffic on VLAN ID 10.	sniff(filter="vlan 10", count=10)
# "mpls"	                                   Capture MPLS traffic.	sniff(filter="mpls", count=10)
# "ip[2:2] > 512"	                           Capture IP packets with total length > 512.	sniff(filter="ip[2:2] > 512", count=10)
# "tcp[tcpflags] & tcp-syn != 0"	           Capture TCP SYN packets only.	sniff(filter="tcp[tcpflags] & tcp-syn != 0", count=10)
# "tcp[13] & 0x02 != 0"	                       Capture SYN flag packets (alternative syntax).	sniff(filter="tcp[13] & 0x02 != 0", count=10)


