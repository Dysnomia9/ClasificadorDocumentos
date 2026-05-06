"""
Generador de documentos sintéticos — 5 clases con vocabulario chileno real.
Clases: factura · contrato · informe · cotizacion · orden_compra
"""
import random, sys
from pathlib import Path
from datetime import datetime, timedelta
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
import pandas as pd
from config import DATA_RAW_DIR, DATA_PROC_DIR, DATASET_CSV, DOCS_POR_CLASE, CLASES, RANDOM_STATE

rng = random.Random(RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════════
# DATOS MAESTROS
# ══════════════════════════════════════════════════════════════════════════

EMPRESAS = [
    "Soluciones Tecnológicas del Sur SpA", "Comercial Austral S.A.",
    "Servicios Integrados Patagonia EIRL", "Consulting & Partners Chile Ltda.",
    "Distribuidora Nacional del Pacífico SpA", "Inversiones Magallanes S.A.",
    "Ingeniería y Proyectos Sur SpA", "Transportes Glaciar Ltda.",
    "Constructora Ventisquero S.A.", "Asesorías del Estrecho EIRL",
    "Importadora Polo Sur SpA", "Manufacturas Andina S.A.",
    "Inmobiliaria Los Canales S.A.", "Agrícola Tierra del Fuego Ltda.",
    "Servicios Portuarios Magallanes SpA", "Minera del Sur S.A.",
    "Pesquera Austral Ltda.", "Clínica Regional del Sur S.A.",
    "Grupo Empresarial Patagónico SpA", "Fundación Cultural Austral",
    "Tecnologías Cóndor SpA", "Comercializadora Los Andes S.A.",
    "Constructora Araucanía EIRL", "Servicios Integrales Valparaíso Ltda.",
    "Holding Empresarial Biobío S.A.", "Desarrollos Digitales Ltda.",
]

PERSONAS = [
    ("Carlos Alberto","Muñoz Sepúlveda"), ("María Fernanda","González Rojas"),
    ("Pedro Ignacio","Vásquez Fuentes"),  ("Ana María","Martínez Díaz"),
    ("Jorge Andrés","Hernández Castro"),  ("Claudia Alejandra","Reyes Morales"),
    ("Luis Felipe","Soto Espinoza"),      ("Patricia Andrea","Flores Cabrera"),
    ("Roberto Esteban","Lagos Navarro"),  ("Daniela Paz","Vargas Poblete"),
    ("Francisco Javier","Pizarro Campos"),("Valentina Isidora","Araya Herrera"),
    ("Sebastián Eduardo","Contreras Leiva"),("Camila Beatriz","Romero Tapia"),
    ("Diego Alonso","Fuentes Molina"),    ("Carolina Paz","Jara Cisternas"),
    ("Rodrigo Andrés","Cáceres Medina"),  ("Paulina Alejandra","Díaz Opazo"),
    ("Felipe Antonio","Morales Ruiz"),    ("Lorena Beatriz","Espinoza Pérez"),
]

CIUDADES   = ["Punta Arenas","Santiago","Valparaíso","Concepción","Puerto Montt",
               "Antofagasta","Iquique","Temuco","Rancagua","Coyhaique",
               "Osorno","La Serena","Arica","Calama","Talca","Chillán"]
COMUNAS    = ["Providencia","Las Condes","Vitacura","Ñuñoa","San Miguel",
               "La Florida","Maipú","Puente Alto","Lo Barnechea","Recoleta"]
MESES      = ["enero","febrero","marzo","abril","mayo","junio",
               "julio","agosto","septiembre","octubre","noviembre","diciembre"]
RUTS_EMP   = ["76.123.456-7","77.234.567-8","78.345.678-9","79.456.789-0",
               "80.567.890-1","81.678.901-2","82.789.012-3","83.890.123-4",
               "76.543.210-K","77.999.888-5","78.001.234-6","79.876.543-2"]
BANCOS_CL  = ["Banco de Chile","BancoEstado","Santander Chile","BCI",
               "Itaú Chile","Scotiabank Chile","Banco Security","BICE"]
REGIONES   = ["Región Metropolitana","Región de Magallanes","Región de Los Lagos",
               "Región del Biobío","Región de Valparaíso","Región de La Araucanía"]

def _rut_p():
    n  = rng.randint(8_000_000, 25_000_000)
    dv = rng.choice("0123456789K")
    return f"{n:,}".replace(",",".")+f"-{dv}"
def _fecha(delta=500):
    d = datetime(2022,1,1)+timedelta(days=rng.randint(0,delta))
    return f"{d.day} de {MESES[d.month-1]} de {d.year}"
def _monto(lo=100_000, hi=5_000_000):
    m = rng.randint(lo//1000, hi//1000)*1000
    return f"${m:,}".replace(",",".")
def _emp():  return rng.choice(EMPRESAS)
def _per():  return rng.choice(PERSONAS)
def _ciu():  return rng.choice(CIUDADES)
def _rute(): return rng.choice(RUTS_EMP)

def _write(path, story):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)

def _S():
    T = ParagraphStyle("T",fontSize=13,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=8)
    N = ParagraphStyle("N",fontSize=9,fontName="Helvetica",spaceAfter=5,leading=14)
    J = ParagraphStyle("J",fontSize=9,fontName="Helvetica",alignment=TA_JUSTIFY,spaceAfter=6,leading=14)
    B = ParagraphStyle("B",fontSize=9,fontName="Helvetica-Bold",spaceAfter=4,spaceBefore=8)
    return T,N,J,B

# ══════════════════════════════════════════════════════════════════════════
# FACTURAS
# ══════════════════════════════════════════════════════════════════════════
TIPOS_FAC = ["FACTURA ELECTRÓNICA","BOLETA DE VENTA ELECTRÓNICA",
             "FACTURA NO AFECTA O EXENTA ELECTRÓNICA",
             "NOTA DE DÉBITO ELECTRÓNICA","NOTA DE CRÉDITO ELECTRÓNICA"]
ITEMS_FAC = [
    ("Consultoría técnica mensual",350_000), ("Licencia software empresarial",480_000),
    ("Mantención preventiva de equipos",120_000), ("Capacitación corporativa",200_000),
    ("Auditoría de sistemas",650_000), ("Instalación y configuración de red",280_000),
    ("Soporte técnico remoto",95_000), ("Desarrollo de módulo a medida",780_000),
    ("Honorarios profesionales",420_000), ("Servicio de transporte de carga",85_000),
    ("Suministro de materiales eléctricos",235_000), ("Arriendo de maquinaria",890_000),
    ("Servicios de aseo industrial",140_000), ("Elaboración de proyecto técnico",560_000),
    ("Venta de insumos de oficina",45_000), ("Asesoría contable mensual",320_000),
    ("Servicio de seguridad privada",580_000), ("Desarrollo sitio web corporativo",960_000),
]

def _generar_factura(path, idx):
    T,N,J,B = _S(); story = []
    tipo  = rng.choice(TIPOS_FAC); folio = rng.randint(10_000,999_999)
    emisor= _emp(); rut_e=_rute(); recept=_emp(); rut_r=_rute()
    ciudad= _ciu(); banco=rng.choice(BANCOS_CL)
    items = rng.sample(ITEMS_FAC, k=rng.randint(2,5))
    cantidades = [rng.randint(1,10) for _ in items]
    story += [Paragraph(tipo,T),
              Paragraph(f"N° {folio}",ParagraphStyle("sf",fontSize=11,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=10)),
              HRFlowable(width="100%",thickness=1.5,color=colors.black),Spacer(1,.3*cm)]
    story += [Paragraph("<b>DATOS DEL EMISOR</b>",B),
              Paragraph(f"Razón Social: {emisor}",N),Paragraph(f"RUT Emisor: {rut_e}",N),
              Paragraph(f"Giro: {rng.choice(['Servicios tecnológicos','Comercio mayorista','Consultoría','Construcción'])}",N),
              Paragraph(f"Dirección: {rng.choice(['Av. Bulnes','Calle Magallanes','Av. España'])} N°{rng.randint(100,9999)}, {ciudad}",N),
              Spacer(1,.2*cm)]
    story += [Paragraph("<b>DATOS DEL RECEPTOR</b>",B),
              Paragraph(f"Razón Social: {recept}",N),Paragraph(f"RUT Receptor: {rut_r}",N),
              Paragraph(f"Fecha de emisión: {_fecha()}",N),Paragraph(f"Vencimiento: {_fecha(30)}",N),
              Spacer(1,.3*cm)]
    tabla_data=[["Descripción","Cant.","P. Unitario","Total"]]
    subtotal=0
    for (desc,pu),cant in zip(items,cantidades):
        tot=pu*cant; subtotal+=tot
        tabla_data.append([desc,str(cant),f"${pu:,}".replace(",","."),f"${tot:,}".replace(",",".")])
    iva=int(subtotal*.19); total=subtotal+iva
    ts=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a2e")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                   ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                   ("ALIGN",(1,0),(-1,-1),"CENTER"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f5")]),
                   ("GRID",(0,0),(-1,-1),.5,colors.grey),("BOTTOMPADDING",(0,0),(-1,-1),4)])
    t=Table(tabla_data,colWidths=[8.5*cm,1.5*cm,3*cm,3*cm]); t.setStyle(ts)
    story.append(t); story.append(Spacer(1,.3*cm))
    tot_data=[["Subtotal Neto:",f"${subtotal:,}".replace(",",".")],
              ["IVA (19%):",f"${iva:,}".replace(",",".")],["TOTAL A PAGAR:",f"${total:,}".replace(",",".")]]
    tt=Table(tot_data,colWidths=[12*cm,4*cm])
    tt.setStyle(TableStyle([("FONTNAME",(0,2),(-1,2),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
                             ("ALIGN",(1,0),(-1,-1),"RIGHT"),("LINEABOVE",(0,2),(-1,2),1,colors.black),
                             ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(tt); story.append(Spacer(1,.4*cm))
    story += [Paragraph(f"<b>Condición de pago:</b> {rng.choice(['Transferencia bancaria','Cheque a 30 días','Contado','Crédito 60 días'])}",N),
              Paragraph(f"<b>Banco:</b> {banco}",N),
              Paragraph("<i>Documento tributario electrónico emitido conforme resolución SII. Timbre Electrónico SII.</i>",
                        ParagraphStyle("pie",fontSize=7,fontName="Helvetica"))]
    _write(path, story)

# ══════════════════════════════════════════════════════════════════════════
# CONTRATOS — 4 subtipos
# ══════════════════════════════════════════════════════════════════════════
def _c_trabajo(path,idx):
    T,N,J,B=_S(); story=[]
    emp=_emp(); rut_e=_rute(); rep_n,rep_a=_per()
    trab_n,trab_a=_per(); rut_t=_rut_p(); ciu=_ciu()
    cargo=rng.choice(["Analista de Sistemas","Técnico en Computación","Ejecutivo Comercial",
                      "Jefe de Operaciones","Contador General","Ingeniero de Proyectos",
                      "Desarrollador de Software","Analista Contable","Coordinador Logístico"])
    sueldo=_monto(500_000,3_000_000)
    jornada=rng.choice(["45 horas semanales de lunes a viernes",
                        "44 horas semanales en turnos rotativos",
                        "40 horas semanales según horario acordado"])
    plazo=rng.choice(["indefinido","a plazo fijo de 6 meses","a plazo fijo de 1 año","por obra o faena"])
    story+=[Paragraph("CONTRATO INDIVIDUAL DE TRABAJO",T),Spacer(1,.3*cm),
            Paragraph(f"En {ciu}, a {_fecha()}, comparecen:",J),Spacer(1,.2*cm)]
    story+=[Paragraph("I. PARTES",B),
            Paragraph(f"<b>EMPLEADOR:</b> {emp}, RUT {rut_e}, representada por {rep_n} {rep_a}, domiciliada en {ciu}.",J),
            Paragraph(f"<b>TRABAJADOR:</b> {trab_n} {trab_a}, RUT {rut_t}, domiciliado/a en {ciu}.",J),Spacer(1,.2*cm)]
    for tc,tx in [
        ("II. CARGO Y FUNCIONES", f"El Empleador contrata al Trabajador para desempeñar el cargo de <b>{cargo}</b>. Las funciones comprenden todas las labores inherentes al puesto."),
        ("III. JORNADA", f"La jornada ordinaria será de <b>{jornada}</b>. Las horas extraordinarias se pagarán con el 50% de recargo conforme al artículo 32 del Código del Trabajo."),
        ("IV. REMUNERACIÓN", f"El Empleador pagará una remuneración mensual bruta de <b>{sueldo}</b>, incluyendo gratificación legal conforme al artículo 50 del Código del Trabajo."),
        ("V. DURACIÓN", f"El contrato de trabajo es <b>{plazo}</b>. Las causales de término se rigen por los artículos 159, 160 y 161 del Código del Trabajo."),
        ("VI. CONFIDENCIALIDAD", f"El Trabajador se obliga a guardar absoluta reserva sobre los negocios, clientes y procesos de la Empresa. Esta obligación rige durante y después de la relación laboral."),
        ("VII. DOMICILIO", f"Para todos los efectos legales las partes fijan domicilio en {ciu} y se someten a los Tribunales de Justicia de Chile."),
    ]:
        story+=[Paragraph(tc,B),Paragraph(tx,J),Spacer(1,.1*cm)]
    story.append(Spacer(1,.8*cm))
    fd=[[f"{rep_n} {rep_a}",f"{trab_n} {trab_a}"],[emp,"El Trabajador"],[f"RUT: {rut_e}",f"RUT: {rut_t}"]]
    ft=Table(fd,colWidths=[9*cm,9*cm])
    ft.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),8),("FONTNAME",(0,0),(-1,1),"Helvetica-Bold")]))
    story.append(ft); _write(path,story)

def _c_arrendamiento(path,idx):
    T,N,J,B=_S(); story=[]
    ador_n,ador_a=_per(); ario_n,ario_a=_per()
    rut_ad=_rut_p(); rut_ar=_rut_p(); ciu=_ciu()
    inmueble=rng.choice([
        f"el inmueble ubicado en calle {rng.choice(['Los Robles','Av. Colón','Errázuriz','Bories'])} N°{rng.randint(100,9999)}, {ciu}",
        f"el departamento N°{rng.randint(1,200)} del edificio en Av. España N°{rng.randint(100,5000)}, {ciu}",
        f"la propiedad con rol de avalúo N°{rng.randint(10000,99999)}-{rng.randint(1,99)}, {ciu}",
        f"la bodega industrial N°{rng.randint(1,50)} del Parque Empresarial de {ciu}",
    ])
    renta=_monto(150_000,800_000); garantia=_monto(150_000,800_000)
    plazo=rng.choice(["12 meses","24 meses","36 meses","mes a mes","6 meses"])
    story+=[Paragraph("CONTRATO DE ARRENDAMIENTO",T),Spacer(1,.3*cm),
            Paragraph(f"En {ciu}, a {_fecha()}, entre las siguientes partes:",J)]
    story+=[Paragraph("PARTES",B),
            Paragraph(f"<b>ARRENDADOR:</b> {ador_n} {ador_a}, RUT {rut_ad}, propietario/a del inmueble, domiciliado/a en {ciu}.",J),
            Paragraph(f"<b>ARRENDATARIO:</b> {ario_n} {ario_a}, RUT {rut_ar}, domiciliado/a en {ciu}.",J),Spacer(1,.2*cm)]
    for tc,tx in [
        ("OBJETO",f"El Arrendador da en arrendamiento a el Arrendatario {inmueble}, destinado a uso {rng.choice(['habitacional','comercial','de oficinas','industrial'])}."),
        ("RENTA Y PAGO",f"La renta mensual es de <b>{renta}</b>, pagadera dentro de los primeros {rng.randint(3,10)} días mediante depósito o transferencia."),
        ("PLAZO",f"El arrendamiento tendrá vigencia de <b>{plazo}</b>. Vencido el plazo, se renueva automáticamente si ninguna parte avisa con 30 días de anticipación."),
        ("GARANTÍA",f"El Arrendatario entrega <b>{garantia}</b> en garantía por daños e incumplimientos, a restituir al término previa revisión del inmueble."),
        ("OBLIGACIONES","El Arrendatario se obliga a conservar el inmueble en buen estado, no subarrendar sin autorización y pagar los servicios básicos."),
        ("RESTITUCIÓN",f"Al término, el Arrendatario restituirá el inmueble conforme a la Ley N°18.101 sobre arrendamiento de predios urbanos."),
    ]:
        story+=[Paragraph(tc,B),Paragraph(tx,J),Spacer(1,.1*cm)]
    story.append(Spacer(1,.8*cm))
    fd=[[f"{ador_n} {ador_a}",f"{ario_n} {ario_a}"],["EL ARRENDADOR","EL ARRENDATARIO"],[f"RUT: {rut_ad}",f"RUT: {rut_ar}"]]
    ft=Table(fd,colWidths=[9*cm,9*cm])
    ft.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),8),("FONTNAME",(0,0),(-1,1),"Helvetica-Bold")]))
    story.append(ft); _write(path,story)

def _c_servicios(path,idx):
    T,N,J,B=_S(); story=[]
    emp=_emp(); rut_e=_rute(); cli=_emp(); rut_c=_rute()
    pr_n,pr_a=_per(); rut_p=_rut_p(); ciu=_ciu()
    serv=rng.choice(["consultoría en gestión empresarial","desarrollo e implementación de software",
                     "asesoría contable y tributaria","servicios de ingeniería y supervisión",
                     "auditoría interna y control de gestión","capacitación y desarrollo organizacional"])
    hon=_monto(400_000,2_000_000); plazo=rng.choice(["3 meses","6 meses","12 meses","por proyecto"])
    story+=[Paragraph("CONTRATO DE PRESTACIÓN DE SERVICIOS PROFESIONALES",T),Spacer(1,.3*cm),
            Paragraph(f"En {ciu}, a {_fecha()}, entre <b>{emp}</b>, RUT {rut_e}, «EL PRESTADOR», y <b>{cli}</b>, RUT {rut_c}, «EL CLIENTE»:",J)]
    for tc,tx in [
        ("PRIMERA: OBJETO",f"EL PRESTADOR se obliga a prestar <b>{serv}</b>, incluyendo todos los entregables para su correcta ejecución."),
        ("SEGUNDA: PLAZO",f"El contrato tendrá vigencia de {plazo} desde su suscripción, renovable por mutuo acuerdo."),
        ("TERCERA: HONORARIOS",f"EL CLIENTE pagará honorarios de <b>{hon}</b> mensuales más IVA, los primeros 5 días hábiles del mes."),
        ("CUARTA: CONFIDENCIALIDAD","EL PRESTADOR mantendrá reserva sobre la información del CLIENTE por 3 años post término."),
        ("QUINTA: TÉRMINO","Cualquier parte puede terminar con 30 días de aviso escrito."),
        ("SEXTA: JURISDICCIÓN",f"Las partes se someten a los Tribunales de {ciu}."),
    ]:
        story+=[Paragraph(tc,B),Paragraph(tx,J),Spacer(1,.1*cm)]
    story.append(Spacer(1,.8*cm)); _write(path,story)

def _c_compraventa(path,idx):
    T,N,J,B=_S(); story=[]
    v_n,v_a=_per(); c_n,c_a=_per()
    rut_v=_rut_p(); rut_c=_rut_p(); ciu=_ciu()
    bien=rng.choice([
        f"vehículo {rng.choice(['Toyota Corolla','Chevrolet Sail','Hyundai Tucson'])} año {rng.randint(2010,2023)}",
        f"bien inmueble en {ciu} con rol SII N°{rng.randint(10000,99999)}-{rng.randint(1,99)}",
        f"maquinaria industrial marca {rng.choice(['Caterpillar','Komatsu','JCB'])} serie N°{rng.randint(100000,999999)}",
        f"equipo computacional compuesto por {rng.randint(5,20)} estaciones de trabajo",
    ])
    precio=_monto(1_000_000,50_000_000)
    story+=[Paragraph("CONTRATO DE COMPRAVENTA",T),Spacer(1,.3*cm),
            Paragraph(f"En {ciu}, a {_fecha()}, entre:",J),
            Paragraph(f"<b>VENDEDOR:</b> {v_n} {v_a}, RUT {rut_v}, domiciliado en {ciu}.",J),
            Paragraph(f"<b>COMPRADOR:</b> {c_n} {c_a}, RUT {rut_c}, domiciliado en {ciu}.",J),Spacer(1,.2*cm)]
    for tc,tx in [
        ("OBJETO",f"El Vendedor vende al Comprador: <b>{bien}</b>. El Comprador declara conocer el estado del bien."),
        ("PRECIO",f"El precio es de <b>{precio}</b>, pagadero {rng.choice(['al contado en este acto','en 3 cuotas iguales','en 6 cuotas mensuales'])}."),
        ("ENTREGA",f"La entrega se efectuará en {ciu} dentro de {rng.randint(3,30)} días de suscrito este instrumento."),
        ("SANEAMIENTO","El Vendedor responde del saneamiento de evicción y vicios ocultos conforme al Código Civil de Chile."),
    ]:
        story+=[Paragraph(tc,B),Paragraph(tx,J),Spacer(1,.1*cm)]
    story.append(Spacer(1,.5*cm)); _write(path,story)

_SUBTIPOS_CONTRATO=[_c_trabajo,_c_trabajo,_c_trabajo,
                    _c_arrendamiento,_c_arrendamiento,_c_arrendamiento,
                    _c_servicios,_c_servicios,_c_compraventa]

def _generar_contrato(path, idx):
    rng.choice(_SUBTIPOS_CONTRATO)(path, idx)

# ══════════════════════════════════════════════════════════════════════════
# INFORMES
# ══════════════════════════════════════════════════════════════════════════
TIPOS_INF=["INFORME DE GESTIÓN","REPORTE TÉCNICO","INFORME DE AVANCE DE PROYECTO",
           "INFORME DE AUDITORÍA INTERNA","REPORTE FINANCIERO TRIMESTRAL",
           "INFORME EJECUTIVO DE RESULTADOS","REPORTE DE INDICADORES KPI",
           "MEMORIA ANUAL DE ACTIVIDADES","INFORME DE RIESGOS OPERACIONALES"]
AREAS_INF=["Gerencia de Operaciones","Departamento de TI","Área de Finanzas",
           "División de Proyectos","Unidad de Control de Gestión","Área de Calidad"]
INDICADORES=[
    ("Cumplimiento del plan operativo","{}%"),("Ejecución presupuestaria","{}% del total asignado"),
    ("Satisfacción de clientes internos","{}/5.0"),("Incidencias resueltas","{} de {} reportadas"),
    ("Tiempo promedio de respuesta","{} horas hábiles"),("Productividad por unidad","{}% respecto a meta"),
]

def _generar_informe(path, idx):
    T,N,J,B=_S(); story=[]
    tipo=rng.choice(TIPOS_INF); emp=_emp(); area=rng.choice(AREAS_INF)
    aut_n,aut_a=_per(); mes=rng.choice(MESES); anio=rng.choice([2022,2023,2024])
    story+=[Paragraph(tipo,T),
            Paragraph(f"Período: {mes.capitalize()} {anio}",ParagraphStyle("s",fontSize=10,alignment=TA_CENTER,spaceAfter=4)),
            Paragraph(f"Organización: {emp}",ParagraphStyle("s2",fontSize=9,alignment=TA_CENTER,spaceAfter=4)),
            Paragraph(f"Elaborado por: {aut_n} {aut_a}  |  {area}",ParagraphStyle("s3",fontSize=9,alignment=TA_CENTER,spaceAfter=10)),
            HRFlowable(width="100%",thickness=1,color=colors.black),Spacer(1,.3*cm)]
    story+=[Paragraph("1. RESUMEN EJECUTIVO",B),
            Paragraph(f"El presente informe expone los resultados del período {mes} {anio}. Se evaluaron los principales KPI y se identificaron áreas de mejora. El cumplimiento de metas alcanzó un {rng.randint(72,98)}%.",J)]
    story+=[Paragraph("2. ANÁLISIS DE RESULTADOS",B)]
    for nombre,fmt in rng.sample(INDICADORES,k=4):
        val = fmt.format(rng.randint(60,99)) if fmt.count("{}")==1 else fmt.format(rng.randint(60,99),rng.randint(100,200))
        story.append(Paragraph(f"• <b>{nombre}:</b> {val}",N))
    story.append(Paragraph(f"El análisis comparativo muestra tendencia {'positiva' if rng.random()>.35 else 'de leve baja'} respecto al período anterior.",J))
    story+=[Paragraph("3. CONCLUSIONES Y RECOMENDACIONES",B)]
    for r in rng.sample(["Fortalecer mecanismos de seguimiento de indicadores en tiempo real.",
                          "Revisar procedimientos conforme a la normativa vigente.",
                          "Implementar plan de capacitación para el personal con brechas detectadas.",
                          "Reforzar controles de auditoría interna en áreas con desviaciones.",
                          "Incorporar herramientas para automatización de reportes y control."],k=3):
        story.append(Paragraph(f"• {r}",N))
    story+=[Spacer(1,.5*cm),
            Paragraph(f"<i>Elaborado el {_fecha()} por {area}. Distribución: dirección, jefaturas y archivo.</i>",
                      ParagraphStyle("pie",fontSize=7,fontName="Helvetica",alignment=TA_CENTER))]
    _write(path,story)

# ══════════════════════════════════════════════════════════════════════════
# COTIZACIONES
# ══════════════════════════════════════════════════════════════════════════
ITEMS_COT=[
    ("Notebook empresarial Intel Core i7 16GB 512GB SSD",980_000),
    ("Switch administrable 24 puertos PoE Gigabit",420_000),
    ("Licencia anual software gestión documental",350_000),
    ("Servicio mantención preventiva equipos TI",180_000),
    ("Access Point WiFi 6 doble banda interior",145_000),
    ("Monitor 27 pulgadas IPS 4K HDMI/DisplayPort",320_000),
    ("UPS 1500VA con protección de red y batería",195_000),
    ("Impresora multifunción color A3 laser",780_000),
    ("Servidor rack Intel Xeon 32GB ECC RAM",2_150_000),
    ("Capacitación TI in-company 8 horas presencial",390_000),
    ("Cámara IP seguridad 4K exterior resistente",125_000),
    ("Silla ergonómica oficina soporte lumbar ajustable",220_000),
    ("Mesa reuniones 8 personas estructura metálica",580_000),
    ("Servicio consultoría gestión proyectos mensual",1_200_000),
    ("Disco SSD empresarial 1TB NVMe M.2",145_000),
    ("Firewall empresarial licencia anual incluida",960_000),
    ("Sistema control acceso biométrico 2 puertas",680_000),
    ("Pizarra interactiva 75 pulgadas táctil 4K",1_450_000),
]
CARGOS_EJ=["Ejecutivo Comercial Senior","Gerente de Ventas","Representante Comercial",
           "Account Manager","Jefe de Ventas Corporativas","Ejecutivo de Cuentas Claves"]

def _generar_cotizacion(path, idx):
    T,N,J,B=_S(); story=[]
    num_cot=f"COT-{rng.randint(2023,2025)}-{rng.randint(1000,9999)}"
    emisor=_emp(); rut_e=_rute(); ciudad=_ciu()
    ej_n,ej_a=_per(); cargo_ej=rng.choice(CARGOS_EJ)
    cliente=_emp(); rut_c=_rute(); cont_n,cont_a=_per()
    banco=rng.choice(BANCOS_CL)
    validez=rng.choice(["15 días corridos","30 días corridos","45 días hábiles","60 días corridos"])
    t_entrega=rng.choice(["5 a 7 días hábiles","10 a 15 días hábiles","3 a 4 semanas","inmediata sujeta a stock"])
    forma_pago=rng.choice(["50% al aprobar, 50% contra entrega",
                           "100% transferencia bancaria previa a despacho",
                           "30 días fecha factura previa aprobación",
                           "Contado con 5% de descuento adicional"])
    items=rng.sample(ITEMS_COT,k=rng.randint(3,6)); cantidades=[rng.randint(1,10) for _ in items]

    story+=[Paragraph("COTIZACIÓN / PRESUPUESTO",T),
            Paragraph(f"N° {num_cot}",ParagraphStyle("sf",fontSize=11,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=8)),
            HRFlowable(width="100%",thickness=1.5,color=colors.black),Spacer(1,.3*cm)]
    story+=[Paragraph("<b>DATOS DEL PROVEEDOR</b>",B),
            Paragraph(f"Empresa: {emisor}",N),Paragraph(f"RUT: {rut_e}",N),
            Paragraph(f"Ejecutivo: {ej_n} {ej_a} — {cargo_ej}",N),
            Paragraph(f"Ciudad: {ciudad}",N),Spacer(1,.2*cm),
            Paragraph("<b>DATOS DEL CLIENTE</b>",B),
            Paragraph(f"Empresa: {cliente}",N),Paragraph(f"RUT: {rut_c}",N),
            Paragraph(f"Atención: {cont_n} {cont_a}",N),
            Paragraph(f"Fecha: {_fecha()}  |  Vigencia: <b>{validez}</b>",N),Spacer(1,.3*cm)]

    story.append(Paragraph("<b>DETALLE DE LA OFERTA COMERCIAL</b>",B))
    tabla_data=[["N°","Descripción","Cant.","P.Unit. (neto)","Total (neto)"]]
    subtotal=0
    for i,((desc,pu),cant) in enumerate(zip(items,cantidades),1):
        pu_var=int(pu*rng.uniform(0.85,1.15)/1000)*1000
        tot=pu_var*cant; subtotal+=tot
        tabla_data.append([str(i),desc,str(cant),f"${pu_var:,}".replace(",","."),f"${tot:,}".replace(",",".")])
    iva=int(subtotal*.19); total=subtotal+iva
    ts=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2c3e50")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                   ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),
                   ("ALIGN",(2,0),(-1,-1),"CENTER"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f5")]),
                   ("GRID",(0,0),(-1,-1),.4,colors.grey),("BOTTOMPADDING",(0,0),(-1,-1),3)])
    t=Table(tabla_data,colWidths=[.8*cm,7.5*cm,1.2*cm,3*cm,3*cm]); t.setStyle(ts)
    story+=[t,Spacer(1,.3*cm)]
    tot_data=[["Subtotal neto:",f"${subtotal:,}".replace(",",".")],
              ["IVA (19%):",f"${iva:,}".replace(",",".")],["TOTAL:",f"${total:,}".replace(",",".")]]
    tt=Table(tot_data,colWidths=[12*cm,3.5*cm])
    tt.setStyle(TableStyle([("FONTNAME",(0,2),(-1,2),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
                             ("ALIGN",(1,0),(-1,-1),"RIGHT"),("LINEABOVE",(0,2),(-1,2),1,colors.black),
                             ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story+=[tt,Spacer(1,.4*cm)]
    story+=[Paragraph("<b>CONDICIONES COMERCIALES</b>",B),
            Paragraph(f"• Forma de pago: {forma_pago}",N),
            Paragraph(f"• Tiempo de entrega: {t_entrega}",N),
            Paragraph(f"• Los valores son en pesos chilenos y <b>no incluyen IVA</b>.",N),
            Paragraph(f"• Cotización válida por <b>{validez}</b> desde su emisión.",N),
            Paragraph(f"• Precios sujetos a disponibilidad de stock al confirmar el pedido.",N),
            Paragraph(f"• No se requiere compromiso para solicitar cotización. Sin compromiso.",N),
            Paragraph(f"• Transferencia a {banco}, cuenta indicada al confirmar.",N),
            Spacer(1,.3*cm),
            Paragraph(f"<i>Agradecemos su consulta. Para aprobar esta cotización responder al ejecutivo {ej_n} {ej_a}. Valores más IVA.</i>",
                      ParagraphStyle("pie",fontSize=8,fontName="Helvetica",leading=12))]
    _write(path,story)

# ══════════════════════════════════════════════════════════════════════════
# ÓRDENES DE COMPRA
# ══════════════════════════════════════════════════════════════════════════
ITEMS_OC=[
    ("Resma papel bond A4 75g 500 hojas",4_200),
    ("Cartucho tóner HP LaserJet compatible",38_000),
    ("Escritorio ejecutivo melanina 160x80cm",185_000),
    ("Silla operativa giratoria sin apoyabrazos",68_000),
    ("Notebook HP ProBook 450 Intel i5 8GB RAM",890_000),
    ("Mouse inalámbrico ergonómico USB",18_500),
    ("Proyector Full HD 3500 lúmenes HDMI VGA",480_000),
    ("Botellón agua purificada 20 litros",4_800),
    ("Cable red UTP Cat6 rollo 305 metros certificado",89_000),
    ("Disco SSD 500GB interno SATA certificado",55_000),
    ("Extintor PQS 6kg ABC certificado SEC",28_000),
    ("Switch no administrable 8 puertos Gigabit",45_000),
    ("Papel fotográfico A4 glossy 200g 100 hojas",12_500),
    ("Teclado español USB resistente salpicaduras",22_000),
    ("Multímetro digital profesional categoría III",65_000),
    ("Archivador palanca oficio tamaño carta",3_500),
    ("Jabón líquido dispensador higiene industrial 5L",8_900),
    ("Cinta adhesiva transparente 48mm x 100m",2_800),
]
AREAS_OC=["Gerencia de Operaciones","Departamento de TI","Área de Finanzas",
          "División de Proyectos","Departamento de RRHH","Área de Infraestructura",
          "Gerencia Comercial","Departamento de Logística","Unidad de Compras"]
CARGOS_COMP=["Jefe de Compras","Gerente de Adquisiciones","Encargado de Abastecimiento",
             "Subgerente de Compras","Coordinador de Adquisiciones"]

def _generar_orden_compra(path, idx):
    T,N,J,B=_S(); story=[]
    num_oc=f"OC-{rng.randint(2023,2025)}-{rng.randint(10000,99999)}"
    comprador=_emp(); rut_comp=_rute(); ciudad=_ciu()
    proveedor=_emp(); rut_prov=_rute()
    sol_n,sol_a=_per(); aut_n,aut_a=_per()
    cargo_aut=rng.choice(CARGOS_COMP); area=rng.choice(AREAS_OC)
    centro_costo=f"CC-{rng.randint(100,999)}-{rng.choice(list('ABCDEFGH'))}"
    banco=rng.choice(BANCOS_CL); fecha_desp=_fecha(30)
    cond_pago=rng.choice(["30 días fecha factura","60 días fecha factura",
                          "Contado al recibir conformidad","45 días fecha factura"])
    lugar_ent=rng.choice([f"Bodega central {ciudad}",f"Oficinas {comprador} — {ciudad}",
                          "Recepción principal edificio matriz","Bodega norte"])
    items=rng.sample(ITEMS_OC,k=rng.randint(2,7)); cantidades=[rng.randint(1,20) for _ in items]

    story+=[Paragraph("ORDEN DE COMPRA",T),
            Paragraph(f"N° {num_oc}",ParagraphStyle("sf",fontSize=11,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=8)),
            HRFlowable(width="100%",thickness=1.5,color=colors.black),Spacer(1,.3*cm)]
    story+=[Paragraph("<b>EMPRESA COMPRADORA</b>",B),
            Paragraph(f"Razón Social: {comprador}",N),Paragraph(f"RUT: {rut_comp}",N),
            Paragraph(f"Ciudad: {ciudad}",N),Paragraph(f"Área solicitante: {area}",N),
            Paragraph(f"Centro de costo: {centro_costo}",N),
            Paragraph(f"Solicitante: {sol_n} {sol_a}",N),Spacer(1,.2*cm),
            Paragraph("<b>PROVEEDOR</b>",B),
            Paragraph(f"Razón Social: {proveedor}",N),Paragraph(f"RUT Proveedor: {rut_prov}",N),
            Paragraph(f"Fecha emisión OC: {_fecha()}  |  Despacho requerido: {fecha_desp}",N),Spacer(1,.3*cm)]

    story.append(Paragraph("<b>DETALLE DE LA ADQUISICIÓN</b>",B))
    tabla_data=[["N°","Código","Descripción","Cant.","P.Unit.","Total"]]
    subtotal=0
    for i,((desc,pu),cant) in enumerate(zip(items,cantidades),1):
        pu_var=int(pu*rng.uniform(0.90,1.10)/100)*100
        tot=pu_var*cant; subtotal+=tot
        cod=f"PRD-{rng.randint(1000,9999)}"
        tabla_data.append([str(i),cod,desc,str(cant),f"${pu_var:,}".replace(",","."),f"${tot:,}".replace(",",".")])
    iva=int(subtotal*.19); total=subtotal+iva
    ts=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16a085")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                   ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),
                   ("ALIGN",(3,0),(-1,-1),"CENTER"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f0faf9")]),
                   ("GRID",(0,0),(-1,-1),.4,colors.grey),("BOTTOMPADDING",(0,0),(-1,-1),3)])
    t=Table(tabla_data,colWidths=[.6*cm,2*cm,6.5*cm,1.2*cm,2.5*cm,2.7*cm]); t.setStyle(ts)
    story+=[t,Spacer(1,.3*cm)]
    tot_data=[["Subtotal neto:",f"${subtotal:,}".replace(",",".")],
              ["IVA (19%):",f"${iva:,}".replace(",",".")],["TOTAL OC:",f"${total:,}".replace(",",".")]]
    tt=Table(tot_data,colWidths=[12*cm,3.5*cm])
    tt.setStyle(TableStyle([("FONTNAME",(0,2),(-1,2),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
                             ("ALIGN",(1,0),(-1,-1),"RIGHT"),("LINEABOVE",(0,2),(-1,2),1,colors.black),
                             ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story+=[tt,Spacer(1,.4*cm)]
    story+=[Paragraph("<b>CONDICIONES DE COMPRA</b>",B),
            Paragraph(f"• Condición de pago: {cond_pago}",N),
            Paragraph(f"• Lugar de entrega: {lugar_ent}",N),
            Paragraph(f"• Emitir factura indicando N° OC: <b>{num_oc}</b>.",N),
            Paragraph(f"• Adjuntar guía de despacho al momento de la entrega en bodega.",N),
            Paragraph(f"• No se aceptan entregas parciales sin autorización previa del área de compras.",N),
            Paragraph(f"• Transferencia a {banco} según datos en la factura.",N),
            Paragraph(f"• El solicitante debe firmar recepción conforme en el sistema.",N),
            Spacer(1,.5*cm),
            Paragraph("<b>AUTORIZACIÓN</b>",B),
            Paragraph(f"Compra autorizada por: {aut_n} {aut_a} — {cargo_aut}",N),
            Spacer(1,.8*cm)]
    fd=[[f"{sol_n} {sol_a}",f"{aut_n} {aut_a}"],["Solicitante",cargo_aut]]
    ft=Table(fd,colWidths=[9*cm,9*cm])
    ft.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("FONTSIZE",(0,0),(-1,-1),8),
                             ("FONTNAME",(0,0),(-1,1),"Helvetica-Bold")]))
    story.append(ft); _write(path,story)

# ══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════
GENERADORES = {
    "factura":      _generar_factura,
    "contrato":     _generar_contrato,
    "informe":      _generar_informe,
    "cotizacion":   _generar_cotizacion,
    "orden_compra": _generar_orden_compra,
}

def generar_dataset(n_por_clase=DOCS_POR_CLASE, verbose=True):
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    registros = []
    for clase in CLASES:
        carpeta = DATA_RAW_DIR / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        for old in carpeta.glob("doc_*.pdf"): old.unlink()
        gen_fn = GENERADORES[clase]
        if verbose: print(f"  → Generando {n_por_clase} docs de clase '{clase}'...")
        for i in range(n_por_clase):
            fname = carpeta / f"doc_{i:04d}.pdf"
            try:
                gen_fn(fname, i)
                registros.append({"path": str(fname), "clase": clase})
            except Exception as e:
                if verbose: print(f"    [WARN] doc_{i:04d}: {e}")
    df = pd.DataFrame(registros)
    df.to_csv(DATASET_CSV, index=False)
    if verbose: print(f"\n✓ Dataset: {len(df)} docs → {DATASET_CSV}")
    return df

if __name__ == "__main__":
    print("Generando dataset con vocabulario chileno real — 5 clases...")
    generar_dataset(verbose=True)
