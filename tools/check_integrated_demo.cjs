const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../demo'),ctx=vm.createContext({window:{}});
for(const file of ['data.js','replays.js','contract-math.js'])vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),ctx);
const {STUDY:D,CONTRACT_REPLAYS:scenes,CONTRACT_MATH:M}=ctx.window;
const rates={fuelTrip:800,driverShift:1250,truckRepairShift:500,loaderActive:1900,loaderRepairShift:500,weigherHour:100,operatorShift:2000,mechanicShift:2000,powerActive:400,kitHour:350};
for(const p of D.predictions)assert.ok(Math.abs(M.cost(p,rates)-p.cost_uah)<1e-6,p.configuration);
for(const c of Object.values(D.controls))assert.ok(Math.abs(M.cost(c,rates)-c.cost_uah)<1e-6);
for(const s of scenes){const p=M.physical(s,s.summary.all_returns_finish_min),cost=M.cost(p,rates);assert.ok(Math.abs(cost-s.expected_cost)<1e-6,s.configuration);assert.ok(Math.abs(s.trips.reduce((a,t)=>a+t.cargo_weight,0)-10000)<1e-6);assert.ok(s.summary.all_returns_finish_min>s.summary.loading_finish_min);assert.equal(p.trips,s.deliveries);const later=M.physical(s,s.summary.all_returns_finish_min+60);assert.equal(M.cost(later,rates),cost);assert.equal(M.cost(M.physical(s,495),rates),0);assert.ok(s.days.filter(d=>d.full_day).every(d=>d.meets_minimum));}
console.log(`PASS: 1836 forecasts, control means, ${scenes.length} full replay costs/mass, independent finish times and frozen final costs.`);
