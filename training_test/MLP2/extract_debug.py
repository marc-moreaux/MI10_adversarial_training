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

all_nlls = []
all_miss = []

if __name__ == "__main__":
    for learning_eps in  [.0, .1, .2, .25, .3]: #[.0, .1, .2, .25, .3]:

        print "\n\nlearning eps is ", learning_eps

        # To store the results
        nlls = []
        miss = []
        val = [0, 1]

        Good_to_go = False
        for line in open("out"+str(learning_eps)+".txt"):

            # Ignore first 200 values
            if Good_to_go is False:
                if re.match("==> Compute adversarial dataset(.)*", line) is not None:
                    Good_to_go = True
                    miss.append(val[0].group('tata'))
                    nlls.append(val[1].group('tata'))
                else:
                    if match("test_y_misclass") :
                        val[0] = match("test_y_misclass")
                    elif match("test_y_nll") :
                        val[1] = match("test_y_nll")

            # Print the nll and misclassified vals
            if Good_to_go:
                val = match("test_y_misclass")
                if val is not None:
                    miss.append(val.group('tata'))

                val = match("test_y_nll")
                if val is not None:
                    nlls.append(val.group('tata'))

        print_result(nlls, miss)
        all_nlls.append(nlls)
        all_miss.append(miss)
