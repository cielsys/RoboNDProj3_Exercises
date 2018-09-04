import matplotlib

# See https://stackoverflow.com/questions/3285193/how-to-switch-backends-in-matplotlib-python
def Backend_Switch(whichBackEnd):
    matplotlib.use(whichBackEnd, warn=False, force=True)
    from matplotlib import pyplot as plt
    print "Switched to:", matplotlib.get_backend()

def BackEnd_List():
    gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
    non_gui_backends = matplotlib.rcsetup.non_interactive_bk
    print ("Non Gui backends are:", non_gui_backends)
    print ("Gui backends I will test for", gui_env)
    for gui in gui_env:
        print ("testing", gui)
        try:
            matplotlib.use(gui,warn=False, force=True)
            from matplotlib import pyplot as plt
            print ("    ",gui, "Is Available")
            plt.plot([1.5,2.0,2.5])
            fig = plt.gcf()
            fig.suptitle(gui)
            plt.show()
            print ("Using ..... ",matplotlib.get_backend())
        except:
            print ("    ",gui, "Not found")

BackEnd_List()