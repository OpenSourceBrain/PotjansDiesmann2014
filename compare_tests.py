import yaml

with open('NEST_SLI/.test.nestsli.mep') as f:
    sli = yaml.safe_load(f)

with open('PyNEST/.test.pynest.mep') as f:
    pynest = yaml.safe_load(f)

with open('PyNN/.test.pynn.mep') as f:
    pynn = yaml.safe_load(f)

rates = []
l = ['-']
for p in ['l23','l4','l5','l6']:
    for b in ['e','i']:
        rates.append('spike_rate_%s%s'%(p,b))
        l.append('%s%s'%(p,b))

sli_r = []
pynest_r = []
pynn_r = []

for r in rates:

    rate = pynn['experiments'][r]['expected']['spike rate']
    pynn_r.append(rate)
    print('PyNN %s: %s'%(r, rate))

    if 'l23e' in r:
        r+='_0'
        
    rate = sli['experiments'][r]['expected']['spike rate']
    sli_r.append(rate)
    print('SLI %s: %s'%(r, rate))
        
    rate = pynest['experiments'][r]['expected']['spike rate']
    pynest_r.append(rate)
    print('PyNEST %s: %s'%(r, rate))


from matplotlib import pyplot as plt

fig, ax = plt.subplots()
plt.title('Spike rates as used in OMV tests')
plt.plot(sli_r, label = "SLI", marker = 'o', linestyle='--')
plt.plot(pynest_r, label = "PyNEST", marker = 'o', linestyle='--')
plt.plot(pynn_r, label = "PyNN", marker = 'o', linestyle='--')

plt.ylabel('rate (spikes/s)')

ax.set_xticklabels(l)

plt.legend()
plt.savefig("TestedRates.png", bbox_inches="tight")

plt.show()

