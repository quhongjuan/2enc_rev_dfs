import meteor

m = meteor.Meteor()
m.compute_score({0:['123'], 1:['12323']}, {0:['123'], 1:['1231231']})
