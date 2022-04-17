for initheight in np.arange(2.5,6,0.1):
  wn = wntr.network.WaterNetworkModel('drive/MyDrive/minitown.inp')
  sim = wntr.sim.EpanetSimulator(wn)
  tank = wn.get_node('TANK')
  tank.init_level = round(initheight,2)
  results = sim.run_sim()

  tank_Head = results.node['head'].loc[:,'TANK']
  pump1 = results.link['status'].loc[:,'PUMP1']
  pump2 = results.link['status'].loc[:,'PUMP2']
  pumpsFlow = results.link['flowrate'].loc[:,'P310'] # remembet to mult x 1000
  tank_flow = results.node['demand'].loc[:,'TANK'] # remember to mult x 1000
  tH = [i for i in tank_Head.iloc]
  p1 = [i for i in pump1.iloc]
  p2 = [i for i in pump2.iloc]
  pQ = [i*1000 for i in pumpsFlow.iloc]
  tQ = [i*1000 for i in tank_flow.iloc]
  dem = [p-t for p,t in zip(pQ,tQ)]
  X_train += [[ps1,ps2,tHs,d] for ps1,ps2,tHs,d in zip(p1,p2,tH,dem)]
  Y_train += tQ
