load "C:/Users/fran_/Documents/Doctorado/Inicios/Marce/NIS/ClasificacionEstructural/7uv0Membrane.pdb", NIS_7UV0
hide everything
select protein_7UV0, NIS_7UV0 and polymer.protein and chain A
select membrane_7UV0, NIS_7UV0 and not polymer.protein
show cartoon, protein_7UV0
color gray80, protein_7UV0
show sticks, membrane_7UV0
color gray65, membrane_7UV0
set stick_radius, 0.08, membrane_7UV0
set transparency, 0.65, membrane_7UV0
set cartoon_transparency, 0.08
set ray_opaque_background, off
bg_color white
set_color color_superficie, [0.25, 0.55, 0.95]
set_color color_core, [0.95, 0.55, 0.10]
set_color color_sitio_activo, [0.90, 0.05, 0.08]

select superficie_1, protein_7UV0 and resi 10+11+13+56+81+82+107+108+110+111+112+113+114+115+119+122+123+124+126+127+130+158+161+163+211+212+213+214+215+216+217+218+220+221+222+224+225+226+227+228+229+230+233+235+237+239+240+267+270+271+272+273+274+275+276+277+279+304+305+306+307+308+309+310+311+312+313+314+315+316+317+318+319+320+321+324+327+330+331+332
select superficie_2, protein_7UV0 and resi 334+335+367+368+369+370+372+373+374+375+376+377+378+379+380+381+382+383+384+385+386+465+466+467+468+469+470+471+472+473+474+475+476+477+478+479+480+481+482+483+510+511+512+513+515+516+517+520+548+549+550+551+552+553+554+555+556+557+558+559+560+561
select superficie, superficie_1 or superficie_2
color color_superficie, superficie
show sticks, superficie
set stick_radius, 0.22, superficie
disable superficie_1 superficie_2

select core_1, protein_7UV0 and resi 12+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+31+32+33+34+35+36+57+58+59+60+61+62+63+64+65+66+68+70+73+74+75+76+77+78+79+80+83+84+85+86+87+88+89+91+92+93+94+95+96+97+98+99+100+101+102+103+104+105+106+109+116+117+118+120+121+125+128+129+131+132+133+134
select core_2, protein_7UV0 and resi 135+136+137+138+139+140+141+142+143+145+146+147+148+149+150+151+152+153+154+155+156+157+159+160+162+164+165+166+167+168+169+170+171+172+173+174+175+176+177+178+179+180+181+182+183+190+191+192+193+194+195+196+197+198+199+200+201+202+203+204+205+206+207+208+209+210+219+223+231+232+234+236+238+241+242+243+244+245+246+247
select core_3, protein_7UV0 and resi 248+249+250+251+252+253+254+255+256+257+258+259+260+261+262+263+264+265+266+268+269+278+280+281+282+283+284+285+286+287+288+289+290+291+292+294+295+296+297+298+299+300+301+302+303+322+323+325+326+328+329+333+336+337+338+339+340+341+342+343+344+345+346+347+348+349+350+351+352+353+354+355+356+357+358+359+360+361+362+363
select core_4, protein_7UV0 and resi 364+365+366+371+387+388+389+390+391+392+393+394+395+396+397+398+399+400+401+402+403+404+405+406+407+408+409+410+411+412+414+415+418+419+420+421+422+423+424+425+426+427+428+429+430+431+432+433+434+435+436+437+438+439+440+441+442+443+444+445+446+447+448+449+450+451+452+453+454+455+456+457+458+459+460+461+462+463+464+514
select core_5, protein_7UV0 and resi 518+519+521+522+523+524+525+526+527+528+529+530+531+532+533+534+535+536+537+538+539+540+541+542+543+544+545+546+547
select core, core_1 or core_2 or core_3 or core_4 or core_5
color color_core, core
show sticks, core
set stick_radius, 0.22, core
disable core_1 core_2 core_3 core_4 core_5

select sitio_activo_1, protein_7UV0 and resi 67+69+71+72+90+144+293+413+416+417
select sitio_activo, sitio_activo_1
color color_sitio_activo, sitio_activo
show sticks, sitio_activo
show spheres, sitio_activo
set sphere_scale, 0.45, sitio_activo
set stick_radius, 0.22, sitio_activo
disable sitio_activo_1

enable superficie
enable core
enable sitio_activo
zoom all, 8
orient

# Legend:
# superficie = blue
# core = orange
# sitio activo = red
# membrane/lipids = translucent gray
# classification source = C:\Users\fran_\Documents\Doctorado\Inicios\Marce\NIS\ClasificacionEstructural\residuos_clasificados_7uv0.csv
# structure source = C:\Users\fran_\Documents\Doctorado\Inicios\Marce\NIS\ClasificacionEstructural\7uv0Membrane.pdb
png "C:/Users/fran_/Documents/Doctorado/Inicios/Marce/PyMOL_coloreo_estructural/vista_coloreo_7UV0.png", width=1800, height=1400, dpi=300, ray=1
save "C:/Users/fran_/Documents/Doctorado/Inicios/Marce/PyMOL_coloreo_estructural/sesion_coloreo_7UV0.pse"
