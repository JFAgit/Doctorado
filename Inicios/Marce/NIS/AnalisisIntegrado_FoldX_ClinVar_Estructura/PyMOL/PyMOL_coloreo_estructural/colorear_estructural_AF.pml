load "C:/Users/fran_/Documents/Doctorado/MarceNIS/FoldX/EstructuraAlphaFold/AF-Q92911model_Repair.pdb", NIS_AF
hide everything
select protein_AF, NIS_AF and polymer.protein and chain A
select membrane_AF, NIS_AF and not polymer.protein
show cartoon, protein_AF
color gray80, protein_AF
show sticks, membrane_AF
color gray65, membrane_AF
set stick_radius, 0.08, membrane_AF
set transparency, 0.65, membrane_AF
set cartoon_transparency, 0.08
set ray_opaque_background, off
bg_color white
set_color color_superficie, [0.25, 0.55, 0.95]
set_color color_core, [0.95, 0.55, 0.10]
set_color color_sitio_activo, [0.90, 0.05, 0.08]

from pymol.cgo import *
from pymol import cmd
membrane_planes = [ALPHA, 0.18, COLOR, 0.55, 0.55, 0.55, BEGIN, TRIANGLES, VERTEX, -57.531, -40.881, 15.200, VERTEX, 49.427, -40.881, 15.200, VERTEX, 49.427, 73.585, 15.200, VERTEX, -57.531, -40.881, 15.200, VERTEX, 49.427, 73.585, 15.200, VERTEX, -57.531, 73.585, 15.200, END, ALPHA, 0.18, COLOR, 0.55, 0.55, 0.55, BEGIN, TRIANGLES, VERTEX, -57.531, -40.881, -15.200, VERTEX, 49.427, 73.585, -15.200, VERTEX, 49.427, -40.881, -15.200, VERTEX, -57.531, -40.881, -15.200, VERTEX, -57.531, 73.585, -15.200, VERTEX, 49.427, 73.585, -15.200, END]
cmd.load_cgo(membrane_planes, 'membrane_planes_AF')

select superficie_1, protein_AF and resi 1+2+3+4+5+6+7+32+33+34+35+36+37+38+39+40+41+42+43+44+45+46+49+51+52+53+78+79+82+83+85+86+88+89+113+114+115+119+122+123+126+127+128+130+131+180+181+182+183+185+206+207+208+210+211+212+213+214+215+216+217+218+219+220+221+222+224+225+226+227+229+230+231+232+233+234+235+236+237+238
select superficie_2, protein_AF and resi 239+241+242+244+245+298+301+302+304+305+306+307+308+309+310+311+313+314+315+316+317+318+321+327+331+332+334+335+366+368+369+370+371+372+373+374+375+376+377+378+379+380+381+382+383+384+385+386+388+389+390+392+393+467+468+469+470+471+472+473+474+475+476+477+478+479+480+481+482+483+484+485+486+487+488+489+490+491+492+493
select superficie_3, protein_AF and resi 494+495+496+497+498+499+500+501+502+503+504+505+506+507+508+509+510+511+512+513+514+515+516+517+518+519+521+522+525+556+557+559+560+574+577+578+595+596+597+598+599+600+601+602+603+604+605+606+607+608+609+610+611+612+613+614+615+616+617+618+619+620+621+622+623+624+625+626+627+628+629+630+631+632+633+634+635+636+637+638
select superficie_4, protein_AF and resi 639+640+641+642+643
select superficie, superficie_1 or superficie_2 or superficie_3 or superficie_4
color color_superficie, superficie
show sticks, superficie
set stick_radius, 0.22, superficie
disable superficie_1 superficie_2 superficie_3 superficie_4

select core_1, protein_AF and resi 8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+31+47+48+50+54+55+56+57+58+59+60+61+62+63+64+65+66+67+68+69+70+71+72+73+74+75+76+77+80+81+84+87+90+91+92+93+94+95+96+97+98+99+100+101+102+103+104+105+106+107+108+109+110+111+112+116+117
select core_2, protein_AF and resi 118+120+121+124+125+129+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+150+151+152+153+154+155+156+157+158+159+160+161+162+163+164+165+166+167+168+169+170+171+172+173+174+175+176+177+178+179+184+186+187+188+189+190+191+192+193+194+195+196+197+198+199+200+201+202+203+204+205+209+223+228+240+243
select core_3, protein_AF and resi 246+247+248+249+250+251+252+253+254+255+256+257+258+259+260+261+262+263+264+265+266+267+268+269+270+271+272+273+274+275+276+277+278+279+280+281+282+283+284+285+286+287+288+289+290+291+292+293+294+295+296+297+299+300+303+312+319+320+322+323+324+325+326+328+329+330+333+336+337+338+339+340+341+342+343+344+345+346+347+348
select core_4, protein_AF and resi 349+350+351+352+353+354+355+356+357+358+359+360+361+362+363+364+365+367+387+391+394+395+396+397+398+399+400+401+402+403+404+405+406+407+408+409+410+411+412+413+414+415+416+417+418+419+420+421+422+423+424+425+426+427+428+429+430+431+432+433+434+435+436+437+438+439+440+441+442+443+444+445+446+447+448+449+450+451+452+453
select core_5, protein_AF and resi 454+455+456+457+458+459+460+461+462+463+464+465+466+520+523+524+526+527+528+529+530+531+532+533+534+535+536+537+538+539+540+541+542+543+544+545+546+547+548+549+550+551+552+553+554+555+558+561+562+563+564+565+566+567+568+569+570+571+572+573+575+576+579+580+581+582+583+584+585+586+587+588+589+590+591+592+593+594
select core, core_1 or core_2 or core_3 or core_4 or core_5
color color_core, core
show sticks, core
set stick_radius, 0.22, core
disable core_1 core_2 core_3 core_4 core_5

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
# classification source = C:\Users\fran_\Documents\Doctorado\Inicios\Marce\NIS\ClasificacionEstructural\residuos_clasificados_AF_Human.csv
# structure source = C:\Users\fran_\Documents\Doctorado\MarceNIS\FoldX\EstructuraAlphaFold\AF-Q92911model_Repair.pdb
png "C:/Users/fran_/Documents/Doctorado/Inicios/Marce/PyMOL_coloreo_estructural/vista_coloreo_AF.png", width=1800, height=1400, dpi=300, ray=1
save "C:/Users/fran_/Documents/Doctorado/Inicios/Marce/PyMOL_coloreo_estructural/sesion_coloreo_AF.pse"
