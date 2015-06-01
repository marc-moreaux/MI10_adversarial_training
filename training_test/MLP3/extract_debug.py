import re


def match(string):
    my_match = re.match("\t"+string+": (?P<tata>[\d\.]+)", line)
    return my_match


def print_result(l_nlls, l_miss):
    print "${0:.4f} \t$&".format(  float(l_nlls[0])       )
    print "${0:.2f}\%\t$&".format((float(l_miss[0]) * 100))
    print "${0:.4f} \t$&".format(  float(l_nlls[1])       )
    print "${0:.2f}\%\t$&".format((float(l_miss[1]) * 100))
    print "${0:.4f} \t$&".format(  float(l_nlls[3])       )
    print "${0:.2f}\%\t$&".format((float(l_miss[3]) * 100))
    print "${0:.4f} \t$&".format(  float(l_nlls[5])       )
    print "${0:.2f}\%\t$&".format((float(l_miss[5]) * 100))
    print "${0:.4f} \t$&".format(  float(l_nlls[7])       )
    print "${0:.2f}\%\t$&".format((float(l_miss[7]) * 100))
    print "${0:.4f} \t$&".format(  float(l_nlls[9])       )
    print "${0:.2f}\%\t$&".format((float(l_miss[9]) * 100))



if __name__ == "__main__":


    path = "/media/marc/SAMSUNG_SD_/"
    for learning_eps in  [.0, .25]: #[.0, .1, .2, .25, .3]:
    
        print "\n\nlearning eps is ", learning_eps

        # To store the results
        nlls = []
        miss = []

        
        with open("out_1_"+str(learning_eps)+".txt") as fp:
            for line in fp:
                if re.match("\ttest_y_misclass"+": (?P<tata>[\d\.]+)", line):
                    miss.append( re.match("\ttest_y_misclass"+": (?P<tata>[\d\.]+)", line).group('tata') )
                
        print min(miss)

        # print_result(nlls, miss)
        # all_nlls.append(nlls)
        # all_miss.append(miss)
