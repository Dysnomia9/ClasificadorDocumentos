"""
Corpus realista — 5 clases con vocabulario chileno auténtico.
factura · contrato · informe · cotizacion · orden_compra
"""
import random, sys
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY
from config import DATA_RAW_DIR, DATASET_CSV, CLASES, RANDOM_STATE

rng = random.Random(RANDOM_STATE + 7)

FRASES_TRAIN = {
"factura": [
    "La presente factura electrónica da cuenta de los servicios prestados conforme a la normativa tributaria del SII.",
    "El monto neto más el impuesto al valor agregado del diecinueve por ciento corresponde al total indicado.",
    "El receptor deberá cancelar el total dentro del plazo de vencimiento estipulado en el documento.",
    "Se emite el presente documento tributario electrónico con folio asignado por resolución del SII.",
    "El emisor certifica que los montos corresponden a los servicios efectivamente entregados al receptor.",
    "La condición de pago convenida es transferencia bancaria a la cuenta corriente del emisor.",
    "El timbre electrónico SII valida la autenticidad de este documento ante el organismo tributario.",
    "El subtotal neto más el IVA calculado sobre el monto neto corresponde al total a pagar.",
    "Los bienes y servicios descritos fueron recibidos a conformidad por el representante del receptor.",
    "La boleta de honorarios electrónica se emite de acuerdo a la normativa vigente del SII.",
    "El descuento comercial aplicado corresponde al convenio vigente entre emisor y receptor.",
    "Se aplica retención de impuesto conforme a la resolución exenta del Servicio de Impuestos Internos.",
    "El giro del emisor corresponde a servicios tecnológicos según registro en el SII.",
    "La razón social del receptor consta en el registro de contribuyentes conforme al RUT indicado.",
    "La factura incluye honorarios profesionales más gastos reembolsables acordados con el cliente.",
],
"contrato": [
    "Las partes acuerdan que el presente instrumento regirá sus relaciones por el plazo estipulado en las cláusulas.",
    "El prestador se obliga a ejecutar los servicios con la diligencia y profesionalismo que corresponde.",
    "En caso de incumplimiento la parte infractora responderá por los perjuicios causados a la contraparte.",
    "El Empleador contrata al Trabajador para desempeñar el cargo convenido conforme al Código del Trabajo.",
    "La jornada ordinaria será de cuarenta y cinco horas semanales distribuidas de lunes a viernes.",
    "La remuneración mensual bruta se pagará dentro de los primeros días hábiles de cada mes.",
    "El Trabajador se obliga a guardar reserva y confidencialidad respecto de los negocios de la empresa.",
    "Las vacaciones anuales del Trabajador se regirán conforme al artículo sesenta y siete del Código del Trabajo.",
    "El Arrendador da en arrendamiento al Arrendatario el inmueble según las condiciones pactadas.",
    "La renta mensual de arrendamiento se fija en el monto acordado entre las partes contratantes.",
    "El bien inmueble objeto del arrendamiento se destinará exclusivamente al uso habitacional convenido.",
    "La cláusula de confidencialidad obliga a ambas partes a no divulgar información durante la vigencia.",
    "El presente contrato podrá modificarse únicamente mediante acuerdo escrito suscrito por ambas partes.",
    "Las causales de término anticipado se encuentran taxativamente señaladas en las cláusulas del contrato.",
    "La jurisdicción y competencia para resolver conflictos corresponde a los Tribunales de Justicia de Chile.",
],
"informe": [
    "Los indicadores analizados muestran una tendencia positiva respecto al período anterior evaluado.",
    "El nivel de cumplimiento de los objetivos estratégicos alcanzó el porcentaje indicado en la tabla.",
    "Se identificaron áreas de mejora que requieren atención prioritaria por parte de la gerencia.",
    "El análisis de los datos permite concluir que la gestión se encuentra dentro de los parámetros.",
    "Las recomendaciones presentadas se basan en el análisis cuantitativo de los indicadores clave.",
    "El equipo técnico elaboró el presente reporte en base a la información consolidada del período.",
    "Los resultados obtenidos superan las metas establecidas en el plan operativo anual vigente.",
    "Los hallazgos identificados serán comunicados a las jefaturas para implementar acciones correctivas.",
    "El índice de satisfacción registra una mejora significativa respecto al trimestre anterior evaluado.",
    "La auditoría interna determinó que los procedimientos se ejecutan conforme a la normativa institucional.",
    "La memoria anual de actividades da cuenta del desempeño organizacional durante el ejercicio.",
    "El seguimiento de indicadores permite verificar el avance respecto a las metas del plan estratégico.",
    "Se concluye que el nivel de eficiencia operacional es consistente con los estándares del sector.",
    "Los datos recopilados permiten establecer una línea base para comparaciones estadísticas futuras.",
    "El resumen ejecutivo sintetiza los principales hallazgos del período de evaluación analizado.",
],
"cotizacion": [
    "La presente cotización tiene validez de treinta días corridos contados desde la fecha de emisión.",
    "Los valores indicados están expresados en pesos chilenos y no incluyen IVA.",
    "Agradecemos su consulta y esperamos que nuestra oferta comercial sea de su pleno interés.",
    "El tiempo de entrega estimado es de cinco a diez días hábiles desde la aprobación del pedido.",
    "Los precios están sujetos a disponibilidad de stock al momento de confirmar el pedido.",
    "La presente oferta comercial está sujeta a modificación sin previo aviso por el proveedor.",
    "Para aprobar esta cotización favor firmar y enviar al ejecutivo comercial asignado.",
    "El descuento por volumen aplica según el convenio comercial vigente entre las partes.",
    "La forma de pago propuesta es cincuenta por ciento al aprobar y el saldo contra entrega.",
    "Esta cotización no constituye compromiso de compraventa hasta su aceptación formal por escrito.",
    "El ejecutivo comercial asignado queda a disposición para aclarar consultas técnicas o comerciales.",
    "Los valores no incluyen instalación salvo que se especifique expresamente en el detalle.",
    "El presupuesto incluye garantía de fabricante según las condiciones del proveedor original.",
    "Cotización elaborada en respuesta a la solicitud de presupuesto enviada por el cliente.",
    "Los precios no consideran despacho a regiones salvo tarifa adicional indicada en condiciones.",
],
"orden_compra": [
    "La presente orden de compra autoriza la adquisición de los bienes detallados en el presente documento.",
    "El proveedor deberá emitir factura indicando el número de esta orden de compra como referencia.",
    "La recepción conforme de los bienes será certificada por el área solicitante en las dependencias.",
    "El despacho debe efectuarse en el plazo indicado a la bodega central de la empresa compradora.",
    "Esta orden de compra ha sido aprobada por el jefe del departamento de adquisiciones y compras.",
    "El pago se efectuará a treinta días fecha factura una vez recepcionados los bienes a conformidad.",
    "Cualquier modificación a esta OC debe ser autorizada por escrito por el área de compras.",
    "El proveedor debe adjuntar guía de despacho junto con los bienes al momento de la entrega.",
    "El centro de costo asignado corresponde al presupuesto aprobado del área solicitante.",
    "En caso de no poder cumplir el plazo de entrega notificar al comprador con cuarenta y ocho horas.",
    "Los productos entregados deben corresponder exactamente a las especificaciones técnicas indicadas.",
    "Esta orden de compra reemplaza cualquier acuerdo verbal previo entre las partes involucradas.",
    "El área de compras coordinará con el proveedor los detalles logísticos del despacho solicitado.",
    "La adquisición fue solicitada por el jefe de área y autorizada por la gerencia correspondiente.",
    "El solicitante debe firmar la recepción conforme en el sistema de gestión al recibir los bienes.",
],
}

FRASES_TEST = {
"factura": [
    "El crédito fiscal asociado puede ser recuperado por el receptor conforme a la normativa tributaria.",
    "La liquidación incorpora los tributos correspondientes al período de prestación del servicio.",
    "Los valores netos se encuentran afectos al impuesto conforme a la ley tributaria vigente en Chile.",
    "El plazo de cancelación opera desde la fecha de emisión consignada en el comprobante tributario.",
    "La nota de crédito electrónica anula el monto facturado según acuerdo entre emisor y receptor.",
    "El número de folio fue asignado automáticamente por el sistema de facturación electrónica del SII.",
    "La operación queda exenta de IVA según la resolución exenta aplicable a este tipo de servicio.",
    "El comprobante debe registrarse conforme a los principios de contabilidad generalmente aceptados.",
],
"contrato": [
    "Ambas partes declaran tener plena capacidad legal para suscribir el presente instrumento contractual.",
    "La vigencia del acuerdo comienza desde la fecha de suscripción por todos los comparecientes.",
    "Las modificaciones al clausulado solo tendrán validez si constan por escrito y son firmadas.",
    "La parte que desee poner término anticipado deberá notificar con la antelación estipulada.",
    "El incumplimiento faculta a la parte afectada a exigir los perjuicios conforme al derecho chileno.",
    "El presente contrato se regirá de conformidad con la legislación de la República de Chile.",
    "El trabajador declara conocer y aceptar el reglamento interno de la empresa empleadora.",
    "El arrendatario se compromete a mantener el inmueble en perfecto estado de conservación.",
],
"informe": [
    "La variación registrada supera el umbral de alerta definido en los parámetros normativos del período.",
    "El panel de indicadores confirma el avance sostenido hacia el logro de los objetivos institucionales.",
    "Los resultados expuestos respaldan la continuidad de la estrategia implementada en el período.",
    "La brecha entre el resultado y la meta requiere una revisión de los procesos operacionales.",
    "Durante el período evaluado se registraron desviaciones menores respecto al plan de trabajo.",
    "Las observaciones del equipo auditor fueron incorporadas en el presente informe de seguimiento.",
    "El análisis de tendencias muestra una mejora sostenida en los índices de desempeño operacional.",
    "El informe técnico describe la metodología aplicada y los resultados obtenidos en la evaluación.",
],
"cotizacion": [
    "El valor unitario ofertado incluye garantía de doce meses según las condiciones del fabricante.",
    "La oferta es válida únicamente para los ítems y cantidades especificadas en el documento.",
    "Ante consultas técnicas contactar directamente al ejecutivo de ventas asignado a su cuenta.",
    "El presupuesto considera los costos de flete y seguro hasta las instalaciones del cliente.",
    "Los precios no incluyen ningún tipo de impuesto ni cargo adicional salvo que se indique.",
    "La empresa se reserva el derecho de modificar precios ante variaciones en el tipo de cambio.",
    "Esta cotización fue elaborada en respuesta a la solicitud de propuesta recibida con fecha.",
    "El servicio posventa y soporte técnico se encuentra descrito en las condiciones comerciales.",
],
"orden_compra": [
    "La orden de compra fue generada automáticamente por el sistema ERP tras la aprobación gerencial.",
    "El proveedor queda notificado de que no se aceptarán entregas parciales salvo autorización previa.",
    "El departamento de compras coordinará la logística de recepción con el área de bodega.",
    "La factura correspondiente a esta OC debe enviarse al correo de cuentas por pagar indicado.",
    "El área requirente validará las especificaciones técnicas antes de autorizar la recepción conforme.",
    "El número de OC debe estar visible en el exterior del embalaje de los productos despachados.",
    "El contrato marco vigente con este proveedor establece las condiciones comerciales de la compra.",
    "La aprobación de esta OC implica la aceptación de los términos y condiciones del comprador.",
],
}

NEUTRAS_TRAIN = [
    "En conformidad con lo establecido se adjunta la documentación correspondiente para su revisión.",
    "La presente comunicación tiene por objeto informar las gestiones realizadas en el período.",
    "Se solicita dar acuse de recibo y proceder conforme a los procedimientos internos vigentes.",
    "El responsable del área revisó la documentación y confirma su conformidad con lo estipulado.",
    "En virtud de lo anterior se solicita tomar nota y proceder conforme a la normativa vigente.",
    "Los antecedentes han sido verificados y se encuentran en orden para continuar con el proceso.",
    "Agradecemos su atención y quedamos a disposición para cualquier consulta adicional.",
]

NEUTRAS_TEST = [
    "La documentación fue preparada conforme a los requerimientos formales del proceso en curso.",
    "Se pone en conocimiento el resultado de las gestiones efectuadas durante el período indicado.",
    "Para mayor información dirigirse al área responsable a través de los canales habituales.",
]

PLANTILLAS_MIXTAS = {
    "factura":      [("cotizacion",   0.30), ("orden_compra", 0.25)],
    "contrato":     [("informe",      0.30), ("orden_compra", 0.25)],
    "informe":      [("contrato",     0.35), ("factura",      0.20)],
    "cotizacion":   [("factura",      0.35), ("orden_compra", 0.25)],
    "orden_compra": [("cotizacion",   0.35), ("factura",      0.25)],
}


def _texto_train_mixto(clase: str, n: int = 14) -> str:
    clase_ajena, frac = rng.choice(PLANTILLAS_MIXTAS[clase])
    n_aj = max(1, int(n * frac)); n_ne = rng.randint(2,3); n_pr = n - n_aj - n_ne
    fs = rng.choices(FRASES_TRAIN[clase], k=max(1,n_pr)) + rng.choices(FRASES_TRAIN[clase_ajena], k=n_aj) + rng.choices(NEUTRAS_TRAIN, k=n_ne)
    rng.shuffle(fs); return " ".join(fs)


def _texto_train_limpio(clase: str, n: int = 16) -> str:
    fs = rng.choices(FRASES_TRAIN[clase], k=n-3) + rng.choices(NEUTRAS_TRAIN, k=3)
    rng.shuffle(fs); return " ".join(fs)


def _texto_train_corto(clase: str) -> str:
    return " ".join(rng.choices(FRASES_TRAIN[clase], k=rng.randint(3,5)))


def _texto_train_confuso(clase: str, n: int = 15) -> str:
    otras = [c for c in CLASES if c != clase]
    clases_ajenas = rng.sample(otras, k=2)
    n_pr = max(4, int(n * 0.45)); n_aj1 = max(2, int(n * 0.25)); n_aj2 = max(2, int(n * 0.20))
    n_ne = max(1, n - n_pr - n_aj1 - n_aj2)
    fs = rng.choices(FRASES_TRAIN[clase], k=n_pr)
    fs += rng.choices(FRASES_TRAIN[clases_ajenas[0]], k=n_aj1)
    fs += rng.choices(FRASES_TRAIN[clases_ajenas[1]], k=n_aj2)
    fs += rng.choices(NEUTRAS_TRAIN + NEUTRAS_TEST, k=n_ne)
    rng.shuffle(fs); return " ".join(fs)


def _texto_test(clase: str, n: int = 12) -> str:
    nn = rng.randint(1,2)
    fs = rng.choices(FRASES_TEST[clase], k=n-nn) + rng.choices(NEUTRAS_TEST, k=nn)
    rng.shuffle(fs); return " ".join(fs)


def _texto_test_ood(clase: str, n: int = 13) -> str:
    otra = rng.choice([c for c in CLASES if c != clase])
    n_pr = max(4, int(n * 0.50)); n_aj = max(2, int(n * 0.35)); n_ne = max(1, n - n_pr - n_aj)
    fs = rng.choices(FRASES_TEST[clase], k=n_pr) + rng.choices(FRASES_TEST[otra], k=n_aj) + rng.choices(NEUTRAS_TEST + NEUTRAS_TRAIN, k=n_ne)
    rng.shuffle(fs)
    return " ".join(fs)


def _aplicar_ruido_superficial(texto: str, intensidad: float = 0.18) -> str:
    toks = texto.split()
    if not toks:
        return texto
    n = max(1, int(len(toks) * intensidad))
    idxs = rng.sample(range(len(toks)), k=min(n, len(toks)))
    for i in idxs:
        t = toks[i]
        r = rng.random()
        if len(t) > 5 and r < 0.45:
            j = rng.randint(1, len(t)-2)
            toks[i] = t[:j] + " " + t[j:]
        elif r < 0.75:
            toks[i] = t.replace("a","@",1) if "a" in t else t.replace("o","0",1)
        else:
            toks[i] = t[:-1] if len(t) > 3 else t
    return " ".join(toks)


def _escribir_pdf(path: Path, texto: str) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    estilo = ParagraphStyle("b", fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=8)
    story = []
    for fr in texto.replace(". ", ".\n").split("\n"):
        fr = fr.strip()
        if fr:
            fr = fr.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            try: story.append(Paragraph(fr, estilo))
            except: pass
        else: story.append(Spacer(1, 0.15*cm))
    if not story: story.append(Paragraph("Documento.", estilo))
    doc.build(story)


def regenerar_corpus_realista(n_por_clase=145, frac_test=0.20, frac_mixtos=0.33, frac_cortos=0.10, frac_confusos=0.15, frac_test_ood=0.35, verbose=True):
    registros = []
    n_te = max(6, int(n_por_clase * frac_test)); n_tr = n_por_clase - n_te
    n_te_ood = max(1, int(n_te * frac_test_ood)); n_te_id = n_te - n_te_ood
    n_mx = int(n_tr * frac_mixtos); n_co = int(n_tr * frac_cortos); n_cf = int(n_tr * frac_confusos); n_li = n_tr - n_mx - n_co - n_cf
    if n_li < 1:
        n_li = 1
        n_mx = max(0, n_mx - 1)
    if verbose:
        print(f"\n  {len(CLASES)} clases | TRAIN {n_tr} (lim={n_li} mix={n_mx} cor={n_co} conf={n_cf}) | TEST {n_te} (iid={n_te_id} ood={n_te_ood})\n")
    for clase in CLASES:
        carpeta = DATA_RAW_DIR / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        if verbose: print(f"  Generando '{clase}'...")
        idx = 0
        for _ in range(n_li):
            p = carpeta/f"doc_{idx:04d}.pdf"; _escribir_pdf(p, _texto_train_limpio(clase)); registros.append({"path":str(p),"clase":clase,"tipo":"limpio","split":"train","source_id":f"src_{idx%5}"}); idx+=1
        for _ in range(n_mx):
            p = carpeta/f"doc_{idx:04d}.pdf"; _escribir_pdf(p, _aplicar_ruido_superficial(_texto_train_mixto(clase), 0.12)); registros.append({"path":str(p),"clase":clase,"tipo":"mixto","split":"train","source_id":f"src_{idx%5}"}); idx+=1
        for _ in range(n_co):
            p = carpeta/f"doc_{idx:04d}.pdf"; _escribir_pdf(p, _aplicar_ruido_superficial(_texto_train_corto(clase), 0.20)); registros.append({"path":str(p),"clase":clase,"tipo":"corto","split":"train","source_id":f"src_{idx%5}"}); idx+=1
        for _ in range(n_cf):
            p = carpeta/f"doc_{idx:04d}.pdf"; _escribir_pdf(p, _aplicar_ruido_superficial(_texto_train_confuso(clase), 0.18)); registros.append({"path":str(p),"clase":clase,"tipo":"confuso","split":"train","source_id":f"src_{idx%5}"}); idx+=1
        for _ in range(n_te_id):
            p = carpeta/f"doc_{idx:04d}.pdf"; _escribir_pdf(p, _aplicar_ruido_superficial(_texto_test(clase), 0.10)); registros.append({"path":str(p),"clase":clase,"tipo":"test_holdout","split":"test","source_id":f"src_{5 + (idx%2)}"}); idx+=1
        for _ in range(n_te_ood):
            p = carpeta/f"doc_{idx:04d}.pdf"; _escribir_pdf(p, _aplicar_ruido_superficial(_texto_test_ood(clase), 0.22)); registros.append({"path":str(p),"clase":clase,"tipo":"test_ood","split":"test_ood","source_id":"src_7"}); idx+=1
    df = pd.DataFrame(registros)
    df.to_csv(DATASET_CSV, index=False)
    if verbose:
        print(f"\n✓ {len(df)} docs → {DATASET_CSV}")
        print(f"  Train: {(df['split']=='train').sum()} | Test-ID: {(df['split']=='test').sum()} | Test-OOD: {(df['split']=='test_ood').sum()}")
        print(df["tipo"].value_counts().to_string())
    return df

if __name__ == "__main__":
    regenerar_corpus_realista(n_por_clase=145, verbose=True)
