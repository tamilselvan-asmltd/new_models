from bertopic import BERTopic
import pandas as pd
import numpy as np
import seaborn as sb


# Replace 'path/to/your/model' with the actual path
merged_model = BERTopic.load("/docker-entrypoint-ddx.d/ddx_agent/resources/Merged_model_09_26_2024")

merged_model.calculate_probabilities=True


import spacy

def clean_text(text):
    # Remove unwanted characters and extra spaces
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = " ".join(lines)
    return cleaned_text

# Original text
EDI830 = """
Planning Schedule with Release Capacity
Seo Set Purpose Code
Ss. Acme
wa Manufacturir Mutually Defined
—_e
— Reference ID
1514893
Start Date
3/16/2024
Carrier
Code: USPN
Ship To
DC #H1A
Code: 043849
Schedule Type Schedule Qty Code Supplier Code
Planned Shipment Based Actual Discrete Quantities 434
Volex Item # Item Description | vom |
430900 3" Widget
Forecast Interval Grouping of Forecast Date Warehouse Loc:
Code Forecast Load ID
Pf 200 | | aizsizo24 |
1 1
"""

# Clean and format the text
EDI830 = clean_text(EDI830)


#print(merged_model.get_topic_info())

topics, probs = merged_model.transform(EDI830)

df_topics=merged_model.get_topic_info()

# Create a DataFrame from the dictionary
df_MM_prob = pd.DataFrame(probs)

print(df_MM_prob)


topics, probs
print(topics, probs)




text_EDI830="""
header Planning Schedule with Release Capacity set_purpose_code Mutually Defined reference_id 1514893 start_date 3/16/2024 carrier Code  USPN ship_to  DC #1A  Code  043849 items Schedule Type Planned Shipment Based Schedule Qty Code Actual Discrete Quantities Supplier Code 434 meta_headers  items Volex Item # 430900 Item Description 3" Widget UOM Each inner Forecast Code Firm Interval Grouping of Forecast  Qty 200 Forecast Date Load 4/23/2024 Warehouse Location ID 1A Forecast Code Planning Interval Grouping of Forecast  Qty 100 Forecast Date Load 4/30/2024 Warehouse Location ID 1A Forecast Code Planning Interval Grouping of Forecast  Qty 400 Forecast Date Load 4/14/2024 Warehouse Location ID 1A Volex Item # 300100 Item Description 5" Sprocket UOM Each inner Forecast Code Firm Interval Grouping of Forecast  Qty 100 Forecast Date Load 4/16/2024 Warehouse Location ID 1A Forecast Code Planning Interval Grouping of Forecast  Qty 200 Forecast Date Load 4/30/2024 Warehouse Location ID 1A
HEADER_ONLY
header set_purpose_code reference_id start_date carrier ship_to Schedule Type Schedule Qty Code Supplier Code meta_headers Volex Item # Item Description UOM Forecast Code Interval Grouping of Forecast Qty Forecast Date Load Warehouse Location ID
"""
text_EDI810="""
reference_id inv00000156 header_items INV00000156 5/1/2024 header Invoice ship_to Acme Inc 123 Main Street Springfield, IL 62701 USA Code   54325 items PO # 2024050 Ship Date 5/08/2024 Vendor # 98098 meta_headers  Total $11024.71 shipping_cost  50 other_charge_amount  10 other_charge_description  Handling fee items Line # #1 Buyer's Part # 043678 Vendor Item # 726683 Description Grainger PO #  0789545 Shipping Cost Description  Freight Charge Quantity 5 UOM Each Price $49.95 Amount $249.75 Line # #2 Buyer's Part # 487137 Vendor Item # 740848 Description Grainger PO #  401612 Shipping Cost Description  Freight Charge Quantity 7 UOM Pound Price $913.61 Amount $6395.27 Line # #3 Buyer's Part # 297140 Vendor Item # 314891 Description Grainger PO #  277729 Shipping Cost Description  Freight Surcharge Quantity 1 UOM Pound Price $857.17 Amount $857.17 Line # #4 Buyer's Part # 018349 Vendor Item # 570772 Description Grainger PO #  191480 Shipping Cost Description  Delivery Charge Quantity 1 UOM Box Price $432.92 Amount $432.92 Line # #5 Buyer's Part # 761203 Vendor Item # 412472 Description Grainger PO #  153002 Shipping Cost Description  Handling Charge Quantity 4 UOM Kilogram Price $757.4 Amount $3029.60 meta_headers
HEADER_ONLY
reference_id header_items header ship_to PO # Ship Date Vendor # meta_headers Total shipping_cost other_charge_amount other_charge_description Line # Buyer's Part # Vendor Item # Description Quantity UOM Price Amount meta_headers

"""

text_EDI850="""

ship_to Acme Inc Code Type  Assigned by Buyer or Buyer's Agent Code   54325 store  Store #  54325 reference_id order_12345 header_items Original New Order Order_12345 5/1/2024 header Purchase Order ship_from Acme Inc Code Type  Assigned by Buyer or Buyer's Agent Code   54325 distribution_center  DC #  54325 items Release # 4783291 Associated PO # 2024050 Status In Process Do Not Ship Before 5/6/2024 Do Not Ship After 5/10/2024 Ship Date 5/08/2024 Requested Delivery 5/12/2024 Routing Sequence Origin Carrier, Shipper's Routing (Rail), Transportation Method Best Way(Shippers Option) SCAC USPS Routing UPS GROUND Service Delivery Ground Backorder No Back Order Currency U.S. Dollars Customer # 54313451 Promotion # Currency Discount items Method of Payment Collect Transportation Terms Cost and Freight Location # Nearest Cross Street Description  meta_headers FOB items Line # 1 UPC # 043000003 524 Vendor Item # 726683 Description  Qty 5 UOM Each Price 49.95 Amount 249 Line # 2 UPC # 683770683 845 Vendor Item # 732780 Description 4" Widget Qty  UOM Each Price 806.32 Amount 6448 Line # 3 UPC # 847144849 841 Vendor Item # 572689 Description 5" Sprocket Qty 3 UOM Meter Price 816.32 Amount  Line # 4 UPC # 485183138 287 Vendor Item # 388158 Description 4" Widget Qty 5 UOM  Price 10.32 Amount 50 Line # 5 UPC # 180897546 468 Vendor Item # 870056 Description 5" Sprocket Qty 10 UOM Meter Price 447.32 Amount 4470 Line # 6 UPC # 321835722 359 Vendor Item # 753816 Description 5" Sprocket Qty 2 UOM Pound Price 247.32 Amount 494 Line # 7 UPC # 009800107 160 Vendor Item # 166898 Description 3" Widget Qty 7 UOM Pound Price  Amount 6307 Line # 8 UPC # 279730344 663 Vendor Item # 386729 Description 3" Widget Qty 4 UOM Meter Price 92.32 Amount 368 Line # 9 UPC # 846641183 275 Vendor Item # 709121 Description  Qty 4 UOM Each Price 866.32 Amount 3464 Line # 10 UPC # 034317256 610 Vendor Item # 196031 Description 4" Widget Qty 7 UOM Pound Price 255.32 Amount 1785 Line # 11 UPC # 316101043 539 Vendor Item # 562273 Description 4" Widget Qty 2 UOM  Price 630.32 Amount 1260 Line # 12 UPC # 556408231 744 Vendor Item # 006674 Description 5" Sprocket Qty 5 UOM Case Price 280.32 Amount 1400 Line # 13 UPC # 777619268 536 Vendor Item # 305119 Description 5" Sprocket Qty 7 UOM Case Price 344.32 Amount 2408 Line # 14 UPC # 461120192 365 Vendor Item # 054151 Description 4" Widget Qty 3 UOM Box Price  Amount 1710 Line # 15 UPC # 331345719 881 Vendor Item # 755015 Description 5" Sprocket Qty 9 UOM Meter Price 551.32 Amount 4959 Line # 16 UPC # 882328419 059 Vendor Item # 201845 Description  Qty 7 UOM Meter Price 831.32 Amount 5817 Line # 17 UPC # 402085880 231 Vendor Item # 557706 Description 3" Widget Qty 5 UOM Pound Price 636.32 Amount 3180 Line # 18 UPC # 225992912 190 Vendor Item # 575230 Description  Qty 9 UOM Kilogram Price  Amount 5310 Line # 19 UPC # 208429863 310 Vendor Item # 879332 Description 3" Widget Qty 3 UOM Foot Price 679.32 Amount  Line # 20 UPC # 906742732 262 Vendor Item # 974395 Description 5" Sprocket Qty 7 UOM Kilogram Price 128.32 Amount 896 meta_headers  Total 55060.75 total_qty  112
HEADER_ONLY
ship_to store reference_id header_items header ship_from distribution_center Release # Associated PO # Status Do Not Ship Before Do Not Ship After Ship Date Requested Delivery Routing Sequence Transportation Method SCAC Routing Service Delivery Backorder Currency Customer # Promotion # Method of Payment Transportation Terms Location # Description meta_headers Line # UPC # Vendor Item # Description Qty UOM Price Amount meta_headers Total total_qty
"""


import PyPDF2
import pdfplumber
from PIL import Image
import pytesseract
import os

# Specify the folder path
folder_path = "/docker-entrypoint-ddx.d/ddx_agent/resources/Generated_EDI_DOC"

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Open the PDF file
        with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
            # Initialize an empty string to store the text
            text = ''

            # Iterate over each page in the PDF
            for page in pdf.pages:
                # Extract the image from the current page
                image = page.to_image(resolution=200)

                # Save the image to a temporary file
                image.save('temp.png', 'PNG')

                # Extract the text from the image using Tesseract-OCR
                text += pytesseract.image_to_string(Image.open('temp.png'))

                # Remove the temporary file
                os.remove('temp.png')

            # Save the extracted text to a file with the same name as the PDF file
            with open(os.path.join(folder_path, filename.replace('.pdf', '.txt')), 'w') as f:
                f.write(text)


import os



# Create an empty dictionary to store the file contents
file_contents = {}

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        # Open the text file and read its contents
        with open(os.path.join(folder_path, filename), 'r') as f:
            # Store the contents in the dictionary
            file_contents[f"text_{filename.replace('.txt', '')}"] = f.read()



print(file_contents)



text_dict = {
  "text_EDI830": text_EDI830,
  "text_EDI810": text_EDI810,
  "text_EDI850": text_EDI850,
  "EDI830": EDI830,
  "text_EDI830_1_2":file_contents['text_EDI830_1_2'],
  "text_EDI830_3":file_contents['text_EDI830_3'],
  "text_EDI830_4":file_contents['text_EDI830_4'],
  "text_EDI830_5":file_contents['text_EDI830_5'],
  "text_EDI850_1":file_contents['text_EDI850_1'],
  "text_EDI850_2":file_contents['text_EDI850_2'],
  "text_EDI850_3":file_contents['text_EDI850_3'],

}


import pandas as pd

# Create a dictionary where each key is a name and each value is a prob list
prob_dict = {name: merged_model.transform(text)[1].tolist() for name, text in text_dict.items()}

# Create a DataFrame from the dictionary
df_prob = pd.DataFrame(prob_dict)

# Apply pd.Series to each column and explode
df_prob = df_prob.apply(pd.Series.explode)

# we can use the following trick to explode all columns at once
df_prob = df_prob.explode(list(df_prob.columns))

# Reset the index
df_prob = df_prob.reset_index(drop=True)




print(df_prob)
df_merged = pd.concat([df_topics, df_prob], axis=1)
print(df_merged)


topic_prob=df_merged.drop(['Topic','Count','Name','CustomName','Representative_Docs', 'PartOfSpeech','Keybert','Representative_Docs'], axis=1)

print(topic_prob)

print(topic_prob)

df_json = topic_prob.to_json(orient='records', indent=4)

# Print the resulting JSON
print(df_json)


topic_prob.to_excel("/docker-entrypoint-ddx.d/ddx_agent/resources/mergedmodel_topic_prob.xlsx", index=False)



Document1="""
Date: September 14, 2024

Due Date: September 28, 2024

Bill To:
John DOE
1982 San Jose AVE
Campbell, 95124, CA
USA

From:
Pure Wafer Inc
4587 ringwood ave
SAn Jose 95130 Ca
USA

Description of Services:

Item	Description	Quantity	Rate	Amount
1	Web Design	1	$500.00	$500.00
2	Content Writing	5	$100.00	$500.00
3	Search Engine Optimization (SEO)	1 month	$250.00	$250.00
Subtotal			$1,250.00
Tax (5%)			$62.50
Total			$1,312.50

"""

#PO
Document2="""
Invoice Number: INV001
Date: September 29, 2024
Billing Information:
Company Name: Meta AI Solutions
Address: 123 Main St, Anytown, USA 12345
Email:
Phone: 555-555-5555
Client Information:
Company Name: BERTopic Testing Inc.
Address: 456 Elm St, Othertown, USA 67890
Email:
Phone: 555-123-4567 """

#Alice in Wonderland
Document3="""Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

Illo2
There was nothing so very remarkable in that, nor did Alice think it so [Pg 4]very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be too late!" But when the Rabbit actually took a watch out of its waistcoat-pocket and looked at it and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and, burning with curiosity, she ran across the field after it and was just in time to see it pop down a large rabbit-hole, under the hedge. In another moment, down went Alice after it!

The rabbit-hole went straight on like a tunnel for some way and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down what seemed to be a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time, as she went down, to look about her. First, she tried to make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and [Pg 5]pictures hung upon pegs. She took down a jar from one of the shelves as she passed. It was labeled "ORANGE MARMALADE," but, to her great disappointment, it was empty; she did not like to drop the jar, so managed to put it into one of the cupboards as she fell past it.

Down, down, down! Would the fall never come to an end? There was nothing else to do, so Alice soon began talking to herself. "Dinah'll miss me very much to-night, I should think!" (Dinah was the cat.) "I hope they'll remember her saucer of milk at tea-time. Dinah, my dear, I wish you were down here with me!" Alice felt that she was dozing off, when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over."""



#EDI 810
Document4= """ISA*01*0000000000*01*0000000000*ZZ*ABCDEFGHIJKLMNO*ZZ*123456789012345*101127*1719*U*00400*000003438*0*P*>
GS*IN*4405197800*999999999*20101205*1710*1320*X*004010VICS
ST*810*1004
BIG*20101204*217224*20101204*P792940
REF*DP*099
REF*IA*99999
N1*ST**92*123
ITD*01*3***0**60
IT1*1*4*EA*8.60**UP*999999330023
IT1*2*2*EA*15.00**UP*999999330115
IT1*3*2*EA*7.30**UP*999999330146
IT1*4*4*EA*17.20**UP*999999330184
IT1*5*8*EA*4.30**UP*999999330320
IT1*6*4*EA*4.30**UP*999999330337
IT1*7*6*EA*1.50**UP*999999330634
IT1*8*6*EA*1.50**UP*999999330641
TDS*21740
CAD*****GTCT**BM*99999
CTT*8
SE*18*1004
GE*1*1320
IEA*1*000001320"""
# EDI 850
Document5="""ISA*01*0000000000*01*0000000000*ZZ*ABCDEFGHIJKLMNO*ZZ*123456789012345*101127*1719*U*00400*000003438*0*P*>
GS*PO*4405197800*999999999*20101127*1719*1421*X*004010VICS
ST*850*000000010
BEG*00*SA*08292233294**20101127*610385385
REF*DP*038
REF*PS*R
ITD*14*3*2**45**46
DTM*002*20101214
PKG*F*68***PALLETIZE SHIPMENT
PKG*F*66***REGULAR
TD5*A*92*P3**SEE XYZ RETAIL ROUTING GUIDE
N1*ST*XYZ RETAIL*9*0003947268292
N3*31875 SOLON RD
N4*SOLON*OH*44139
PO1*1*120*EA*9.25*TE*CB*065322-117*PR*RO*VN*AB3542
PID*F****SMALL WIDGET
PO4*4*4*EA*PLT94**3*LR*15*CT
PO1*2*220*EA*13.79*TE*CB*066850-116*PR*RO*VN*RD5322
PID*F****MEDIUM WIDGET
PO4*2*2*EA
PO1*3*126*EA*10.99*TE*CB*060733-110*PR*RO*VN*XY5266
PID*F****LARGE WIDGET
PO4*6*1*EA*PLT94**3*LR*12*CT
PO1*4*76*EA*4.35*TE*CB*065308-116*PR*RO*VN*VX2332
PID*F****NANO WIDGET
PO4*4*4*EA*PLT94**6*LR*19*CT
PO1*5*72*EA*7.5*TE*CB*065374-118*PR*RO*VN*RV0524
PID*F****BLUE WIDGET
PO4*4*4*EA
PO1*6*696*EA*9.55*TE*CB*067504-118*PR*RO*VN*DX1875
PID*F****ORANGE WIDGET
PO4*6*6*EA*PLT94**3*LR*10*CT
CTT*6
AMT*1*13045.94
SE*33*000000010
GE*1*1421
IEA*1*000003438"""


docs = [Document1, Document2, Document3, Document4, Document5]
Topic1, probs1 = merged_model.transform(docs)




# Creating a dataframe with topic assignments and probabilities
df = pd.DataFrame(probs1, columns=[f"Topic_{i+1}" for i in range(probs1.shape[1])])
df['Assigned_Topic'] = Topic1

# Adding a column for Document identifiers
df['Document'] = [f'Document_{i+1}' for i in range(len(Topic1))]

# Reorder the columns to have Document at the front
df = df[['Document'] + [col for col in df.columns if col != 'Document']]


print(df)

df_json = df.to_json(orient='records', indent=4)

# Print the resulting JSON
print(df_json)


# Step 1: Creating a dataframe with topic assignments and probabilities
df = pd.DataFrame(probs1, columns=[f"Topic_{i+1}" for i in range(probs1.shape[1])])
df['Assigned_Topic'] = Topic1  # Assuming Topic1 has the topic number assignments

# Adding a column for Document identifiers
df['Document'] = [f'Document_{i+1}' for i in range(len(Topic1))]

# Step 2: Retrieve topic information (topic numbers and names)
topic_info = merged_model.get_topic_info()

# Step 3: Create a mapping of topic numbers to topic names (or representative words)
topic_mapping = topic_info.set_index('Topic')['Name'].to_dict()  # Assuming 'Name' column contains the topic names

# Step 4: Map the topic names to the Assigned_Topic column
df['Topic_Name'] = df['Assigned_Topic'].map(topic_mapping)

# Step 5: Reorder the columns to have Document at the front
df = df[['Document', 'Assigned_Topic', 'Topic_Name'] + [col for col in df.columns if col not in ['Document', 'Assigned_Topic', 'Topic_Name']]]

# Display the updated dataframe
print(df)

df_json = df.to_json(orient='records', indent=4)

# Print the resulting JSON
print(df_json)
