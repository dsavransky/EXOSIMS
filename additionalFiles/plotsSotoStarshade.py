# Stick these lines after line 155 (before return in generate_dVMap) in SotoStarshade to plot the example missions. Un/comment sections as needed based on the mission being simulated
#        import matplotlib.pyplot as plt
#        plt.rcParams.update({'font.size': 20})
#        from matplotlib.colors import LogNorm
#
#        LON = (np.arange(-70,90,20)*u.deg).to('rad')
#        LAT = (np.arange(-180,180,20)*u.deg).to('rad')
#        lat,lon = np.meshgrid(LAT,LON)
#        fig = plt.figure(figsize=(8, 6))
#        ax = fig.add_subplot(111, projection="mollweide")
#        p = ax.scatter(lat, lon, s=20, c=TL.int_comp)
#        ax.grid(True)
#        # DRO
#        ax.plot(np.array([0,-140])*np.pi/180, np.array([10,-30])*np.pi/180, linewidth=3,label='Slew 1')
#        ax.plot(np.array([-140,-180])*np.pi/180, np.array([-30,30])*np.pi/180, linewidth=3,label='Slew 2')
#        ax.plot(np.array([-180,80])*np.pi/180, np.array([30,-70])*np.pi/180, linewidth=3,label='Slew 3')
#        ax.scatter(0*np.pi/180,10*np.pi/180,s=500,marker='*',c='c',label='Start')
#        ax.scatter(80*np.pi/180,-70*np.pi/180,s=300,marker='8',c='r',label='End')
#        ax.tick_params(axis='x',pad=100)

#        # EML1
##        ax.plot(np.array([160,140])*np.pi/180, np.array([-50,-30])*np.pi/180, linewidth=3,label='Slew 1')
##        ax.plot(np.array([140,100])*np.pi/180, np.array([-30,-30])*np.pi/180, linewidth=3,label='Slew 2')
##        ax.plot(np.array([100,60])*np.pi/180, np.array([-30,-30])*np.pi/180, linewidth=3,label='Slew 3')
##        ax.scatter(160*np.pi/180,-50*np.pi/180,s=500,marker='*',c='c',label='Start')
##        ax.scatter(60*np.pi/180,-30*np.pi/180,s=300,marker='8',c='r',label='End')
##        l_array = np.arange(-150,180,30)

#        # EML2
##        ax.plot(np.array([60,-40])*np.pi/180, np.array([-70,-70])*np.pi/180, linewidth=3,label='Slew 1')
##        ax.plot(np.array([-40,-120])*np.pi/180, np.array([-70,-70])*np.pi/180, linewidth=3,label='Slew 2')
##        ax.plot(np.array([-120,-140])*np.pi/180, np.array([-70,-50])*np.pi/180, linewidth=3,label='Slew 3')
##        ax.scatter(60*np.pi/180,-70*np.pi/180,s=500,marker='*',c='c',label='Start')
##        ax.scatter(-140*np.pi/180,-50*np.pi/180,s=300,marker='8',c='r',label='End')
##        l_array = np.arange(-210,120,30)
#
#        labels = [item.get_text() for item in ax.get_xticklabels()]
#        labels = [str(element)+"$^\circ$" for element in l_array]
#        ax.set_xticklabels(labels)
#        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#        plt.subplots_adjust(right=.78)
#        plt.show()
