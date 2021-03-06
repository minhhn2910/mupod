import numpy as np
#contains all bitwidth reported for 1% and 5% accuracy drop for DNNs
#simply run : python list_bitwidth.py to list all bitwidth
#due to copying-paste the large amount of numbers, if there is error in putting bitwidth incorrectly, please open pull request, I will update
class DNN:
    def __init__(self, int_part, input, mac, frac_in, frac_mac, weight_bw, baseline= [], name="DNN"):
        self.name = name #name of DNN + accuracy
        self.int_part = np.copy(int_part) #integer Bitwidth
        self.input = np.copy(input)#number of input elements in each layers to process 1 image, usually divided by some factorto avoid big number
        self.mac = np.copy(mac) #number of input elements in each layers to process 1 image, usually divided by some factorto avoid big number
        self.frac_in = np.copy(frac_in) #fractional bitwidth for the objective "optimize for input bandwidth"
        self.frac_mac = np.copy(frac_mac) #fractional bitwidth for the objective "optimize for mac energy"
        self.baseline = np.copy(baseline) #baseline bitwidth to compare with (optional)
        self.weight = np.copy(weight_bw) #weight bitwidth for compute power (optional)
    def __str__(self):
        return self.name
    def print_bitwidth(self):
        print ("\nBitwidth of %s :"%(self.name))
        print ("---------------------------")
        print ("Optimized for input bandwidth")
        print (list(self.frac_in + self.int_part)) #list printed out better than np array
        print (" Effective_BW for input: %.2f"%(np.dot(self.frac_in + self.int_part,self.input)/sum(self.input)))
        print (" Effective_BW for MAC: %.2f" %(np.dot(self.frac_in + self.int_part,self.mac)/sum(self.mac)))
        print ("\n---------------------------")
        print ("Optimized for MAC")
        print (list(self.frac_mac + self.int_part))
        print (" Effective_BW for input: %.2f" %(np.dot(self.frac_mac + self.int_part,self.input)/sum(self.input)))
        print (" Effective_BW for MAC: %.2f" %(np.dot(self.frac_mac + self.int_part,self.mac)/sum(self.mac)))
        print ("")


dnns = [] #store all DNNs

#resnet50

int_part = np.array([9, 7, 7, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6, 5, 5, 6, 6, 8, 7, 7, 6, 7, 7, 6, 6, 6, 6, 7, 6, 6, 7, 6, 6, 7, 6, 6, 7, 7, 6, 7, 7, 7, 5, 6, 9, 5, 5, 9, 6, 6, 8])
input = np.array([150528.0, 200704.0, 200704.0, 200704.0, 200704.0, 802816.0, 200704.0, 200704.0, 802816.0, 200704.0, 200704.0, 802816.0, 802816.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 401408.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 200704.0, 25088.0, 25088.0, 100352.0, 25088.0, 25088.0, 100352.0, 25088.0, 25088.0, 2048.0])
mac = np.array([2360280.0, 1027604.0, 256902.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 2055200.0, 513802.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 2055200.0, 513802.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 2055200.0, 513802.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 1027604.0, 2312120.0, 1027604.0, 40960.0])
frac_in = np.array([0, 4, 3, 4, 4, 2, 2, 2, 2, 1, 1, 2, 0, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 0, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 3, 3, 0, 2, 2, 0, 2, 2, 6])
frac_mac = np.array([-1, 4, 5, 3, 4, 3, 2, 2, 3, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 0, 1, 1, 5])
stripes = np.array([9]*len(frac_mac))
weight_bw = 9
resnet50_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "resnet50-1%")
dnns.append(resnet50_1)

#5%
#in
frac_in = np.array([-1, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 0, 1, 1, 1, 2, 2, 0, 1, 1, 0, -1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 1, 5])
#mac
frac_mac = np.array([-2, 3, 4, 2, 3, 2, 1, 1, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, -1, 1, 1, -1, 1, 1, 4])
stripes = np.array([8]*len(frac_mac))
weight_bw = 8

resnet50_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "resnet50-5%")
dnns.append(resnet50_5)


#vgg19
int_part = np.array([9, 11, 13, 14, 15, 15, 15, 15, 17, 17, 17, 16, 15, 14, 14, 12])
input = np.array([150528.0, 3211260.0, 802816.0, 1605630.0, 401408.0, 802816.0, 802816.0, 802816.0, 200704.0, 401408.0, 401408.0, 401408.0, 100352.0, 100352.0, 100352.0, 100352.0])#, 25088.0, 4096.0, 4096.])
mac = np.array([867041.0, 18496900.0, 9248440.0, 18496900.0, 9248440.0, 18496900.0, 18496900.0, 18496900.0, 9248440.0, 18496900.0, 18496900.0, 18496900.0, 4624220.0, 4624220.0, 4624220.0, 4624220.0])#, 1027600.0, 167772.0, 40960.0])
#1%
frac_in = np.array([-1, -5, -5, -6, -7, -7, -8, -9, -8, -9, -9, -8, -6, -6, -4, -4])
frac_mac = np.array([0, -4, -5, -6, -7, -8, -8, -9, -9, -10, -10, -9, -7, -6, -5, -4])
stripes = np.array([9,9,9,8,12,10,10,12,13,11,12,13,13,13,13,13])
weight_bw = 11
vgg19_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "vgg19-1%")
dnns.append(vgg19_1)

#-------------
#5%
#input
frac_in = np.array([-2, -6, -6, -7, -7, -8, -9, -9, -9, -9, -9, -9, -8, -7, -6, -5])
#MAC
# round_safe
frac_mac =np.array([-1, -5, -6, -7, -7, -8, -9, -9, -9, -10, -10, -9, -8, -7, -6, -5])
stripes = np.array([7]*len(frac_mac))
weight_bw = 9

vgg19_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "vgg19-5%")
dnns.append(vgg19_5)


##googlenet

int_part = np.array([ 9,  9, 10,  9,  9, 11,  9, 10,  9, 11, 11, 11, 11, 11, 11, 11, 11,12, 11, 11, 11, 12, 12, 12, 12, 11, 12, 12, 12, 11, 12, 12, 12, 11,
                        11, 11, 11, 11, 11, 11, 11, 10, 11, 10, 11, 10, 10, 10, 10, 10, 10,
                        10, 10,  9, 10,  9, 10])
input = np.array([150528.0, 200704.0, 200704.0, 150528.0, 150528.0, 75264.0, 150528.0, 12544.0, 150528.0, 200704.0, 200704.0, 100352.0, 200704.0, 25088.0, 200704.0, 94080.0, 94080.0, 18816.0, 94080.0, 3136.0, 94080.0, 100352.0, 100352.0, 21952.0, 100352.0, 4704.0, 100352.0, 100352.0, 100352.0, 25088.0, 100352.0, 4704.0, 100352.0, 100352.0, 100352.0, 28224.0, 100352.0, 6272.0, 100352.0, 103488.0, 103488.0, 31360.0, 103488.0, 6272.0, 103488.0, 40768.0, 40768.0, 7840.0, 40768.0, 1568.0, 40768.0, 40768.0, 40768.0, 9408.0, 40768.0, 2352.0, 40768.0])
mac = np.array([1180140.0, 128451.0, 3468170.0, 96337.9, 144507.0, 867041.0, 24084.5, 100352.0, 48169.0, 256901.0, 256901.0, 1734080.0, 64225.3, 602112.0, 128451.0, 180634.0, 90316.8, 352236.0, 15052.8, 37632.0, 60211.2, 160563.0, 112394.0, 442552.0, 24084.5, 75264.0, 64225.3, 128451.0, 128451.0, 578028.0, 24084.5, 75264.0, 64225.3, 112394.0, 144507.0, 731566.0, 32112.6, 100352.0, 64225.3, 264929.0, 165581.0, 903168.0, 33116.2, 200704.0, 132465.0, 104366.0, 65228.8, 225792.0, 13045.8, 50176.0, 52183.0, 156549.0, 78274.6, 325140.0, 19568.6, 75264.0, 52183.0])
#1%
frac_in = np.array([-1, -2, -3, -3, -4, -4, -5, -5, -4, -4, -5, -4, -5, -5, -5, -4, -4, -5, -5, -5, -5, -3, -4, -3, -5, -4, -5, -3, -4, -4, -4, -4, -5, -3, -3, -4, -4, -4, -4, -3, -3, -3, -3, -4, -3, -1, -1, -3, -2, -3, -3, -2, -1, -1, -2, -2, -4])
frac_mac = np.array([-2, -2, -4, -2, -3, -4, -4, -5, -4, -4, -4, -5, -4, -5, -5, -3, -3, -5, -4, -5, -5, -3, -3, -4, -4, -4, -5, -3, -3, -4, -3, -4, -5, -3, -3, -4, -3, -4, -4, -3, -3, -3, -3, -4, -3, -1, -1, -3, -1, -3, -3, -2, -1, -2, -1, -2, -4])
stripes = np.array([10,8,8,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8, 9,9,9,9,9,9, 10,10,10,10,10,10, 8,8,8,8,8,8, 9,9,9,9,9,9, 10,10,10,10,10,10, 8,8,8,8,8,8 ])

googlenet_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "googlenet-1%")
dnns.append(googlenet_1)

#5%
#in
frac_in = np.array([-2, -4, -4, -4, -5, -5, -6, -6, -5, -5, -6, -5, -6, -6, -5, -5, -5, -5, -7, -6, -6, -4, -5, -5, -5, -6, -6, -4, -5, -5, -6, -5, -6, -4, -4, -4, -5, -5, -5, -3, -4, -4, -4, -4, -4, -2, -3, -3, -3, -4, -3, -3, -2, -3, -3, -3, -4])
#mac
frac_mac = np.array([-3, -3, -5, -4, -4, -5, -5, -6, -5, -5, -5, -6, -5, -6, -5, -5, -5, -5, -5, -6, -6, -4, -5, -5, -4, -6, -6, -4, -4, -5, -4, -5, -6, -4, -4, -5, -4, -5, -5, -3, -4, -4, -3, -4, -4, -2, -2, -4, -2, -4, -3, -3, -2, -3, -2, -3, -4])
stripes = np.array([6]*len(frac_mac))
weight_bw = 8

googlenet_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "googlenet-5%")
dnns.append(googlenet_5)



###
#NiN

int_part = np.array([9, 13, 13, 13, 14, 14, 13, 14, 13, 12, 13, 12])
input = np.array([301056.0, 559872.0, 559872.0, 139968.0, 373248.0, 373248.0, 86528.0, 129792.0, 129792.0, 27648.0, 73728.0, 73728.0])
mac = np.array([1016170.0, 268739.0, 268739.0, 4478980.0, 477757.0, 477757.0, 1495200.0, 249201.0, 249201.0, 1274020.0, 377487.0, 368640.0])
#3.5%
#In
frac_in =np.array([-3, -4, -5, -5, -7, -7, -6, -6, -5, -4, -6, -4])
#MAC
frac_mac =np.array([-3, -3, -4, -6, -6, -6, -7, -6, -5, -6, -6, -4])
stripes = np.array([8,8,7,9,7,8,8,9,9,8,7,8])
weight_bw = 10

NiN_35 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "NiN_3.5%")
dnns.append(NiN_35)

#5%
#input
frac_in = np.array([-3, -5, -5, -6, -7, -7, -6, -7, -6, -4, -6, -4])
#mac
frac_mac = np.array([-3, -3, -4, -6, -7, -7, -8, -6, -5, -6, -6, -4])
stripes = np.array([8]*len(frac_mac))
weight_bw = 8

NiN_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "NiN_5%")
dnns.append(NiN_5)

#resnet152

int_part = np.array([9, 7, 7, 6, 5, 7, 6, 5, 7, 5, 5, 7, 7, 5, 6, 7, 5, 5, 7, 5, 5, 7, 5, 6, 7, 5, 5, 7, 5, 5, 7, 5, 6, 7, 6, 6, 7, 7, 5, 5, 7, 5, 6, 7, 7, 7, 8, 8, 8, 8, 8, 7, 9, 8, 8, 9, 8, 9, 10, 9, 9, 10, 9, 8, 10, 8, 8, 10, 9, 10, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 10, 11, 9, 9, 11, 9, 9, 11, 9, 10, 12, 9, 10, 12, 10, 10, 12, 9, 10, 12, 10, 10, 12, 9, 9, 12, 9, 10, 11, 9, 9, 11, 9, 10, 11, 9, 9, 11, 9, 10, 11, 9, 9, 11, 9, 10, 11, 9, 9, 10, 9, 9, 10, 8, 9, 10, 7, 8, 9, 9, 6, 5, 8, 5, 5, 9, 6, 6, 7])
input = np.array([150528.0, 200704.0, 200704.0, 200704.0, 200704.0, 802816.0, 200704.0, 200704.0, 802816.0, 200704.0, 200704.0, 802816.0, 802816.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 100352.0, 100352.0, 401408.0, 401408.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 50176.0, 50176.0, 200704.0, 200704.0, 25088.0, 25088.0, 100352.0, 25088.0, 25088.0, 100352.0, 25088.0, 25088.0, 2048.0])

mac = np.array([4720560.0, 2055208.0, 513804.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 4110400.0, 1027604.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 4110400.0, 1027604.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 4110400.0, 1027604.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 2055208.0, 4624240.0, 2055208.0, 81920.0])

#1%
frac_in = np.array([0, 5, 3, 4, 4, 3, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2, 1, 1, 2, 1, 2, 2, 3, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 3, 2, 1, 2, 2, 1, 2, 2, 6])
frac_mac =np.array([0, 5, 4, 4, 5, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 2, 0, 1, 2, 1, 2, 2, 3, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 5])
stripes = np.array([12]*len(frac_mac))
weight_bw = 11

resnet152_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "resnet152_1%")
dnns.append(resnet152_1)


#5% input
frac_in =np.array([-1, 4, 2, 3, 3, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 0, 1, 1, 0, 1, 1, 2, 2, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, -1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 1, 5])
#mac_count
frac_mac = np.array([-1, 4, 3, 2, 4, 3, 2, 2, 3, 1, 2, 3, 2, 2, 2, 1, 0, 1, 1, 0, 1, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, -1, 0, 1, -1, 1, 1, 0, 1, 1, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 1, 4])
stripes = np.array([11]*len(frac_mac))
weight_bw = 8

resnet152_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "resnet152_5%")
dnns.append(resnet152_5)



#mobilenet
int_part =np.array([3, 4, 5, 5, 5, 4, 5, 5, 4, 3, 4, 3, 4, 3, 3, 2, 4, 2, 4, 3, 5, 5, 6, 5, 6, 3, 4, 6])

input = np.array([150528.0, 401408.0, 401408.0, 802816.0, 200704.0, 401408.0, 401408.0, 401408.0, 100352.0, 200704.0, 200704.0, 200704.0, 50176.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 100352.0, 25088.0, 50176.0, 50176.0, 1024.0] )
mac = np.array ([1.0838e+07, 3.61267e+06, 2.56901e+07, 1.80634e+06, 2.56901e+07, 3.61267e+06, 5.13802e+07, 903168, 2.56901e+07, 1.80634e+06, 5.13802e+07, 451584, 2.56901e+07, 903168, 5.13802e+07, 903168, 5.13802e+07, 903168, 5.13802e+07, 903168, 5.13802e+07, 903168, 5.13802e+07, 225792, 2.56901e+07, 451584, 5.13802e+07, 1.024e+06])

frac_in = np.array([6, 7, 5, 5, 5, 6, 4, 4, 5, 6, 4, 5, 6, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 6, 7, 6, 7])
frac_mac = np.array([6, 9, 6, 6, 5, 7, 4, 6, 5, 7, 4, 8, 5, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 9, 5, 9, 5, 6])
stripes = np.array([10]*len(frac_mac))
weight_bw = 10

mobilenet_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "mobilenet_1%")
dnns.append(mobilenet_1)

#5%
frac_in = np.array([5, 7, 4, 4, 5, 5, 3, 4, 4, 5, 3, 4, 5, 6, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 6, 5, 6])
frac_mac = np.array([5, 8, 5, 6, 5, 6, 3, 6, 4, 6, 3, 7, 4, 7, 3, 7, 3, 7, 3, 6, 3, 6, 3, 8, 4, 8, 4, 5])
stripes = np.array([9]*len(frac_mac))
weight_bw = 9

mobilenet_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "mobilenet_5%")
dnns.append(mobilenet_5)

#squeezenet
#1%
int_part = np.array([9, 12, 13, 13, 13, 13, 13, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12])
input = np.array([154587.0, 290400.0, 48400.0, 48400.0, 387200.0, 48400.0, 48400.0, 387200.0, 96800.0, 96800.0, 186624.0, 23328.0, 23328.0, 186624.0, 34992.0, 34992.0, 279936.0, 34992.0, 34992.0, 279936.0, 46656.0, 46656.0, 86528.0, 10816.0, 10816.0, 86528.0])
mac = np.array([1.73874e+08, 4.6464e+06, 3.0976e+06, 2.78784e+07, 6.1952e+06, 3.0976e+06, 2.78784e+07, 1.23904e+07, 1.23904e+07, 1.11514e+08, 5.97197e+06, 2.98598e+06, 2.68739e+07, 8.95795e+06, 6.71846e+06, 6.04662e+07, 1.34369e+07, 6.71846e+06, 6.04662e+07, 1.79159e+07, 1.19439e+07, 1.07495e+08, 5.53779e+06, 2.7689e+06, 2.49201e+07, 1.152e+08])
frac_in = np.array([-1, -2, -2, -3, -4, -3, -4, -5, -4, -4, -4, -3, -4, -5, -4, -4, -4, -2, -3, -1, -2, -3, -3, -3, -3, -4])
frac_mac = np.array([-2, -1, -1, -3, -3, -3, -4, -4, -4, -5, -3, -3, -5, -4, -4, -5, -4, -2, -4, -1, -2, -4, -2, -3, -4, -5])
stripes = np.array([9]*len(frac_mac))
weight_bw = 8

squeezenet_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "squeezenet_1%")
dnns.append(squeezenet_1)

#5%
frac_in = np.array([-2, -3, -3, -4, -5, -4, -5, -6, -5, -5, -5, -4, -5, -6, -4, -5, -5, -4, -4, -2, -3, -4, -4, -3, -4, -5])
frac_mac = np.array([-3, -2, -2, -4, -3, -3, -5, -4, -4, -6, -4, -4, -6, -4, -4, -6, -4, -3, -5, -1, -3, -5, -3, -3, -5, -6])
stripes = np.array([9]*len(frac_mac))
weight_bw = 7

squeezenet_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "squeezenet_5%")
dnns.append(squeezenet_5)


#alexnet
int_part = np.array([9, 9, 9, 10, 10])#, 10, 8, 7])
input = np.array([154587.0, 69984.0, 43264.0, 64896.0, 64896.0])#, 9216.0#, 4096.0, 4096.0])
mac = np.array([1.05415e+08, 2.23949e+08, 1.4952e+08, 1.1214e+08, 7.47602e+07])
#1%
frac_in =np.array([-3, -3, -4, -4, -3]) # %0.00728116
frac_mac =np.array([-2, -4, -4, -4, -3])
stripes = np.array([9,7,4,5,7])
weight_bw = 10

alexnet_1 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "alexnet_1%")
dnns.append(alexnet_1)

#5%

frac_in =np.array([-4, -5, -5, -6, -5])
frac_mac =np.array([-3, -5, -6, -6, -5])
stripes = np.array([5]*len(frac_mac))
weight_bw = 8

alexnet_5 = DNN(int_part, input, mac, frac_in, frac_mac, weight_bw, stripes, "alexnet_5%")
dnns.append(alexnet_5)

for dnn in dnns:
    print (dnn.name)
    dnn.print_bitwidth()
