#!/usr/bin/env python3

'''
Creating the Constants which can be used within a directory down
in the src folder
'''
CONSTANTS =  {'ORIGINALMANDI' : 'data/original/wholesaleoniondata.csv',
              'ORIGINALRETAIL': 'data/original/retailoniondata.csv',
              'ORIGINALMANDIINFO': 'data/original/mandis.csv',
              'MANDIPRICES'   : 'data/timeseries/mandiPrices.csv',
              'MANDIARRIVALS' : 'data/timeseries/mandiArrivals.csv',
              'RETAILPRICES'  : 'data/timeseries/centresPrices.csv',
              'MANDIPRICES_WHITENED'   : 'data/riteshwhitened/mandiPrices.csv',
              'MANDIARRIVALS_WHITENED' : 'data/riteshwhitened/mandiArrivals.csv',
              'RETAILPRICES_WHITENED'  : 'data/riteshwhitened/centresPrices.csv',
              'MANDIPRICES_UNEXTRAPOLATED'   : 'data/unextrapolated/mandiPrices.csv',
              'MANDIARRIVALS_UNEXTRAPOLATED' : 'data/unextrapolated/mandiArrivals.csv',
              'RETAILPRICES_UNEXTRAPOLATED'  : 'data/unextrapolated/centresPrices.csv',
              'ANOMALIES_NEWSPAPER'  : 'data/anomaly/anomalies.csv',
              'ANOMALIES_ICA'  : 'data/anomaly/anomalies_ica.csv',
              'CENTRESIDRITESH'     : [10, 16, 40, 44, 50],
              'MANDIIDSRITESH'      : [[182, 174, 194],
                                 [281, 404, 351, 312, 165, 70, 293, 164, 407, 166],
                                 [545, 323, 405, 584, 278, 288],
                                 [156, 427],
                                 [279, 376]],
              'CENTRESID'     : [3, 10, 16, 40, 44, 50],
              'CENTRENAMES'   : ['AHEMDABAD','BHUBANESHWAR', 'DELHI', 'LUCKNOW', 'MUMBAI', 'PATNA'],
              'MANDIIDS'      : [[31], [194], [70, 164], [278], [427], [279]],
              'DAYS'          : 3461,
              'ANOMALYNAMES' : ['Hoarding','Inflation','Weather','Fuel','Transport'],
              'STARTDATE'     : '2006-01-01',
              'ENDDATEOLD'       : '2015-06-23',
              'ENDDATE'       : '2017-11-30'
              }
