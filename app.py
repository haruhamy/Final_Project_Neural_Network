import os
import numpy as np
import streamlit as st
from keras.preprocessing import image as keras_image
from keras.models import load_model
from PIL import Image
import time

# Page configuration with dark theme
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="",
    layout="wide"
)

# Custom CSS for dark theme
st.html("""
<style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    .title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .sidebar-section {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: left;
    }
    .result-section {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    p {
        color: white;
        font-size: 1.2rem;
    }
    .result-text {
        padding: 10px;
        border-radius: 5px;
        background-color: #2E2E2E;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #AAAAAA;
        font-size: 0.8rem;
    }
    .stButton>button {
        border: none;
    }
    button {
        background-color: rgb(85 87 84);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    button:hover {
        opacity: 0.8;
        outline: none;
        background-color: rgb(85 87 84);
    }
    button:active {
        background-color: #518622 !important; 
        transform: scale(0.95);
        outline: none;
       
    }
    .stApp {
        # background-color: #2a3937;
    }
    
    
</style>
""")

# Title
st.html("<h1 class='title'>Công cụ phân loại rau củ quả</h1>")

# Load model
@st.cache_resource
def load_my_model():
    return load_model('mobilenet_model.h5')

try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Define classes
label_nutrition = { 
     0: {
        "name": "Đậu",
        "calories": 31,
        "protein": 2.0,
        "fat": 0.2,
        "fiber": 3.4,
        "carbohydrates": 7.0,
        "sugar": 0.2,
        "description": """Đậu rất giàu protein thực vật, tốt cho tim mạch và hệ tiêu hóa. 
        Chứa các axit amin thiết yếu giúp tái tạo cơ bắp. 
        Hàm lượng chất xơ cao giúp cải thiện sức khỏe đường ruột. 
        Cung cấp sắt, giúp ngăn ngừa thiếu máu. 
        Ít chất béo và calo, hỗ trợ kiểm soát cân nặng. 
        Chứa chất chống oxy hóa giúp giảm viêm. 
        Hỗ trợ duy trì lượng đường trong máu ổn định. 
        Giúp giảm cholesterol xấu (LDL). 
        Cung cấp năng lượng bền vững cho cơ thể. 
        Cải thiện chức năng trao đổi chất tổng thể."""
    },
    1: {
        "name": "Mướp đắng (Khổ qua)",
        "calories": 17,
        "protein": 1.0,
        "fat": 0.2,
        "fiber": 2.6,
        "carbohydrates": 3.7,
        "sugar": 1.2,
        "description": """Mướp đắng giúp kiểm soát đường huyết và tăng cường hệ miễn dịch. 
        Giàu vitamin C, giúp tăng cường sức đề kháng. 
        Chứa các hợp chất hỗ trợ tiết insulin tự nhiên. 
        Hỗ trợ tiêu hóa và bảo vệ đường ruột. 
        Giảm viêm và hỗ trợ phục hồi cơ thể. 
        Giúp giảm cân nhờ khả năng giảm hấp thụ chất béo. 
        Cải thiện sức khỏe làn da, giúp giảm mụn. 
        Thanh lọc gan và thận, hỗ trợ thải độc tố. 
        Giúp giảm cholesterol xấu, bảo vệ tim mạch. 
        Tăng cường thị lực nhờ hàm lượng vitamin A cao."""
    },
    2: {
        "name": "Bầu",
        "calories": 15,
        "protein": 0.6,
        "fat": 0.1,
        "fiber": 1.2,
        "carbohydrates": 3.7,
        "sugar": 1.3,
        "description": """Bầu giúp làm mát cơ thể và hỗ trợ tiêu hóa. 
        Giàu nước, giúp cơ thể duy trì độ ẩm tốt. 
        Hỗ trợ giải độc gan và thận. 
        Cung cấp chất xơ giúp ngăn ngừa táo bón. 
        Giúp kiểm soát huyết áp nhờ lượng kali cao. 
        Ít calo, thích hợp cho chế độ ăn kiêng. 
        Chứa chất chống oxy hóa giúp ngăn ngừa lão hóa. 
        Hỗ trợ giảm căng thẳng và thư giãn thần kinh. 
        Tăng cường hệ miễn dịch nhờ vitamin C. 
        Hỗ trợ giảm đau khớp và viêm nhiễm."""
    },
    3: {
        "name": "Cà",
        "calories": 25,
        "protein": 1.0,
        "fat": 0.2,
        "fiber": 3.0,
        "carbohydrates": 6.0,
        "sugar": 2.7,
        "description": """Cà Cung cấp nhiều chất xơ, giúp hỗ trợ tiêu hóa và ngăn ngừa táo bón.
        Chứa lượng nhỏ protein thực vật, giúp cung cấp năng lượng cho cơ thể.
        Giàu chất chống oxy hóa, giúp bảo vệ tế bào khỏi tổn thương.
        Hỗ trợ sức khỏe tim mạch nhờ hàm lượng kali giúp kiểm soát huyết áp.
        Có lợi cho hệ vi khuẩn đường ruột khi được lên men tự nhiên.
        Chứa vitamin nhóm B, giúp duy trì hệ thần kinh khỏe mạnh.
        Hỗ trợ hấp thu sắt nhờ vitamin C, giúp ngăn ngừa thiếu máu.
        Ít calo, phù hợp với người muốn kiểm soát cân nặng.
        Giúp kích thích vị giác, tăng cảm giác ngon miệng trong bữa ăn."""
    },
    4: {
        "name": "Bông cải xanh",
        "calories": 55,
        "protein": 3.7,
        "fat": 0.6,
        "fiber": 3.8,
        "carbohydrates": 11.0,
        "sugar": 2.2,
        "description": """Bông cải xanh giàu vitamin C, chất xơ và chất chống oxy hóa. 
        Giúp tăng cường hệ miễn dịch và bảo vệ tế bào. 
        Hỗ trợ sức khỏe tim mạch bằng cách giảm cholesterol. 
        Tăng cường sức khỏe xương nhờ hàm lượng canxi và vitamin K. 
        Giúp kiểm soát cân nặng nhờ lượng calo thấp. 
        Bảo vệ thị lực và giảm nguy cơ mắc bệnh mắt. 
        Hỗ trợ tiêu hóa và ngăn ngừa táo bón. 
        Chứa sulforaphane giúp ngăn ngừa ung thư. 
        Giúp giải độc gan và thanh lọc cơ thể. 
        Cung cấp năng lượng bền vững và giảm căng thẳng."""
    },
    5: {
        "name": "Bắp cải",
        "calories": 25,
        "protein": 1.3,
        "fat": 0.1,
        "fiber": 2.5,
        "carbohydrates": 5.8,
        "sugar": 3.2,
        "description": """Bắp cải giúp giảm viêm và hỗ trợ tiêu hóa. 
        Giàu vitamin C, giúp tăng cường hệ miễn dịch. 
        Cung cấp chất chống oxy hóa bảo vệ tế bào khỏi tổn thương. 
        Giúp giảm nguy cơ mắc bệnh tim mạch. 
        Chứa hợp chất sulforaphane giúp ngăn ngừa ung thư. 
        Hỗ trợ kiểm soát cân nặng nhờ lượng calo thấp. 
        Giúp cải thiện sức khỏe đường ruột và ngăn ngừa táo bón. 
        Cung cấp vitamin K giúp xương chắc khỏe. 
        Tăng cường quá trình giải độc cơ thể. 
        Giúp duy trì làn da khỏe mạnh và chống lão hóa."""
    },
    6: {
        "name": "Ớt chuông",
        "calories": 40,
        "protein": 1.2,
        "fat": 0.3,
        "fiber": 1.7,
        "carbohydrates": 9.0,
        "sugar": 5.0,
        "description": """Ớt chuông giàu vitamin C giúp tăng cường hệ miễn dịch. 
        Hỗ trợ sức khỏe tim mạch nhờ hàm lượng kali và flavonoid. 
        Cung cấp chất chống oxy hóa giúp bảo vệ da và mắt. 
        Giúp tăng cường trao đổi chất, hỗ trợ giảm cân. 
        Hỗ trợ kiểm soát huyết áp và cải thiện lưu thông máu. 
        Giảm nguy cơ viêm nhiễm nhờ hàm lượng quercetin cao. 
        Cải thiện chức năng thần kinh và bảo vệ não bộ. 
        Hỗ trợ tiêu hóa và giúp hấp thụ chất dinh dưỡng tốt hơn. 
        Giảm căng thẳng và cải thiện tâm trạng nhờ vitamin B6. 
        Chống lão hóa và giúp duy trì làn da sáng khỏe."""
    },
    7: {
        "name": "Cà rốt",
        "calories": 41,
        "protein": 0.9,
        "fat": 0.2,
        "fiber": 2.8,
        "carbohydrates": 9.6,
        "sugar": 4.7,
        "description": """Cà rốt giàu beta-carotene, tốt cho mắt và da. 
        Giúp tăng cường hệ miễn dịch nhờ hàm lượng vitamin A cao. 
        Hỗ trợ tiêu hóa và giảm nguy cơ táo bón. 
        Giúp kiểm soát huyết áp nhờ kali. 
        Cung cấp chất chống oxy hóa giúp giảm viêm. 
        Giúp giảm cholesterol và bảo vệ tim mạch. 
        Hỗ trợ kiểm soát cân nặng và đốt cháy mỡ thừa. 
        Giúp làm đẹp da và làm chậm quá trình lão hóa. 
        Hỗ trợ ngăn ngừa ung thư nhờ hợp chất thực vật. 
        Tăng cường trí nhớ và sức khỏe não bộ."""
    },
    8: {
        "name": "Bông cải trắng",
        "calories": 25,
        "protein": 1.9,
        "fat": 0.3,
        "fiber": 2.0,
        "carbohydrates": 5.0,
        "sugar": 1.9,
        "description": """Súp lơ trắng giàu chất xơ và vitamin K, tốt cho xương. 
        Giúp giảm viêm và hỗ trợ tiêu hóa. 
        Cung cấp chất chống oxy hóa giúp bảo vệ tế bào. 
        Hỗ trợ giảm nguy cơ mắc bệnh tim mạch. 
        Giúp duy trì cân nặng nhờ lượng calo thấp. 
        Hỗ trợ giải độc cơ thể nhờ các hợp chất lưu huỳnh. 
        Giúp tăng cường trí nhớ và bảo vệ não bộ. 
        Hỗ trợ kiểm soát đường huyết và giảm nguy cơ tiểu đường. 
        Giúp làm đẹp da và tóc nhờ vitamin C. 
        Cung cấp năng lượng và tăng cường sức khỏe tổng thể."""
    },
    9: {
        "name": "Dưa chuột (Dưa leo)",
        "calories": 16,
        "protein": 0.7,
        "fat": 0.1,
        "fiber": 0.5,
        "carbohydrates": 3.6,
        "sugar": 1.7,
        "description": """Dưa leo giúp cung cấp nước và duy trì độ ẩm cho cơ thể. 
        Hỗ trợ làm đẹp da và giảm quầng thâm mắt. 
        Giúp kiểm soát cân nặng nhờ lượng calo thấp. 
        Cung cấp chất chống oxy hóa giúp giảm viêm. 
        Hỗ trợ tiêu hóa và giảm nguy cơ táo bón. 
        Giúp thanh lọc cơ thể và đào thải độc tố. 
        Hỗ trợ kiểm soát huyết áp nhờ kali. 
        Giúp làm mát cơ thể trong mùa nóng. 
        Cải thiện sức khỏe tóc và móng. 
        Hỗ trợ sức khỏe tim mạch và tuần hoàn máu."""
    },
    10: {
        "name": "Đu đủ",
        "calories": 43,
        "protein": 0.5,
        "fat": 0.2,
        "fiber": 1.7,
        "carbohydrates": 11.0,
        "sugar": 8.0,
        "description": """Đu đủ hỗ trợ tiêu hóa và giảm táo bón nhờ enzyme papain. 
        Giàu vitamin A, giúp bảo vệ mắt và cải thiện thị lực. 
        Hỗ trợ làm đẹp da và làm chậm quá trình lão hóa. 
        Giúp tăng cường hệ miễn dịch nhờ vitamin C. 
        Hỗ trợ giảm viêm và đau khớp. 
        Giúp duy trì cân nặng ổn định. 
        Tăng cường sức khỏe tim mạch và giảm cholesterol. 
        Giúp kiểm soát huyết áp nhờ kali. 
        Hỗ trợ giải độc gan và cải thiện chức năng gan. 
        Tăng cường năng lượng và cải thiện tâm trạng."""
    },
    11: {
        "name": "Khoai tây",
        "calories": 77,
        "protein": 2.0,
        "fat": 0.1,
        "fiber": 2.2,
        "carbohydrates": 17.0,
        "sugar": 0.8,
        "description": """Khoai tây cung cấp năng lượng dồi dào nhờ hàm lượng carbohydrate cao.
        Hỗ trợ sức khỏe tim mạch nhờ kali giúp kiểm soát huyết áp.
        Giàu vitamin C giúp tăng cường hệ miễn dịch.
        Chứa chất xơ giúp hỗ trợ tiêu hóa và giảm nguy cơ táo bón.
        Hỗ trợ làm đẹp da nhờ chất chống oxy hóa.
        Giúp cải thiện sức khỏe não bộ và giảm căng thẳng.
        Cung cấp năng lượng bền vững cho người tập luyện thể thao.
        Hỗ trợ kiểm soát cân nặng nếu chế biến hợp lý.
        Giúp ngăn ngừa bệnh thiếu máu nhờ hàm lượng sắt và folate.
        Cung cấp vitamin B6 giúp tăng cường chức năng thần kinh."""
    },
    12: {
        "name": "Bí ngô",
        "calories": 26,
        "protein": 1.0,
        "fat": 0.1,
        "fiber": 0.5,
        "carbohydrates": 6.5,
        "sugar": 2.8,
        "description": """Bí ngô giàu beta-carotene giúp bảo vệ mắt và cải thiện thị lực.
        Hỗ trợ sức khỏe tim mạch nhờ kali và chất xơ.
        Giúp tăng cường hệ miễn dịch nhờ vitamin C.
        Hỗ trợ tiêu hóa và giúp đường ruột khỏe mạnh.
        Cung cấp chất chống oxy hóa giúp bảo vệ tế bào khỏi lão hóa.
        Giúp kiểm soát cân nặng nhờ lượng calo thấp.
        Hỗ trợ làm đẹp da và tóc nhờ vitamin A và E.
        Giúp giảm viêm và hỗ trợ xương chắc khỏe.
        Cung cấp năng lượng cho cơ thể mà không gây tăng cân.
        Hỗ trợ giảm căng thẳng và cải thiện giấc ngủ."""
    },
    13: {
        "name": "Củ cải trắng",
        "calories": 16,
        "protein": 0.6,
        "fat": 0.1,
        "fiber": 1.6,
        "carbohydrates": 3.4,
        "sugar": 1.9,
        "description": """Củ cải trắng giúp thanh lọc cơ thể và hỗ trợ giải độc gan.
        Hỗ trợ hệ tiêu hóa và giúp giảm táo bón.
        Giúp kiểm soát huyết áp nhờ kali.
        Giàu vitamin C giúp tăng cường hệ miễn dịch.
        Hỗ trợ giảm viêm và chống oxy hóa mạnh.
        Giúp duy trì cân nặng nhờ lượng calo thấp.
        Tăng cường sức khỏe da và làm sáng da.
        Hỗ trợ giảm nguy cơ mắc bệnh tiểu đường.
        Giúp kiểm soát cholesterol và bảo vệ tim mạch.
        Cung cấp năng lượng tự nhiên giúp giảm mệt mỏi."""
    },
    14: {
        "name": "Cà chua",
        "calories": 18,
        "protein": 0.9,
        "fat": 0.2,
        "fiber": 1.2,
        "carbohydrates": 3.9,
        "sugar": 2.6,
        "description": """Cà chua giàu lycopene giúp bảo vệ da và làm chậm lão hóa.
        Hỗ trợ sức khỏe tim mạch và giảm cholesterol.
        Giúp tăng cường hệ miễn dịch nhờ vitamin C.
        Hỗ trợ tiêu hóa và ngăn ngừa táo bón nhờ chất xơ.
        Giúp kiểm soát huyết áp nhờ kali.
        Cung cấp chất chống oxy hóa giúp bảo vệ tế bào khỏi tổn thương.
        Giúp giảm viêm và hỗ trợ sức khỏe xương.
        Hỗ trợ kiểm soát cân nặng nhờ lượng calo thấp.
        Giúp cải thiện trí nhớ và giảm nguy cơ suy giảm nhận thức.
        Cung cấp nước giúp cơ thể luôn tươi trẻ và tràn đầy năng lượng."""
    }
}

description = []
# Main layout
col1, col2, col3 = st.columns([2, 3, 4])

with col1:
    st.html("""<div class='sidebar-section'>
    <h2>Hướng dẫn sử dụng.</h2>
    <p>1. Tải ảnh lên hình ảnh để phân loại (JPG only).</p>
    <p>2. Chờ model phân tích hình ảnh.</p>
    <p>3. Kết quả sau khi phân tích sẽ hiển thị ở bên phải.</p>
    <p>Note: những rau củ quả có thể phân loại gồm:
    Đậu, Mướp đắng, Bầu, Cà tím, Bông cải xanh, Bắp cải, Ớt chuông, Cà rốt, Súp lơ, Dưa chuột, Đu đủ, Khoai tây, Bí ngô, Củ cải, Cà chua.</p>    
    </p>
    """)

with col2:
    # Only accept JPG files
    uploaded_file = st.file_uploader("chọn hình ảnh...", type=["jpg"], label_visibility="collapsed")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="ảnh tải lên", use_container_width =True)
    

with col3:
    st.html("""<p style="border-bottom:1px solid #7f7f7f">Kết quả phân loại</p>""")
    
    result_placeholder = st.empty()
    
    if uploaded_file is not None and model_loaded:
        with st.spinner('đang phân tích hình ảnh...'):
            time.sleep(1)
            
            test_image = img.resize((224, 224))
            test_image = keras_image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            
            result = model.predict(test_image)
            pred_labels = np.argmax(result, axis=1)
            result_placeholder.html(f"""
                <div class='result-container'>
                    <h3 class='result-title'>Name: {label_nutrition.get(pred_labels[0])["name"]}</h3>
                    <p>Thành phần dinh dưỡng có trong 100 gram {label_nutrition.get(pred_labels[0])["name"]}:</p>
                    <p><strong>Calories:</strong> {label_nutrition.get(pred_labels[0])["calories"]} kcal</p>
                    <p><strong>Carbohydrates:</strong> {label_nutrition.get(pred_labels[0])["carbohydrates"]} g</p>
                    <p><strong>Protein:</strong> {label_nutrition.get(pred_labels[0])["protein"]} g</p>
                    <p><strong>Chất béo:</strong> {label_nutrition.get(pred_labels[0])["fat"]} g</p>
                    <p><strong>chất xơ:</strong> {label_nutrition.get(pred_labels[0])["fiber"]} g</p>
                    <p><strong>Mô tả:</strong> {label_nutrition.get(pred_labels[0])["description"]}</p>
                </div>
            """)

            if st.button("🔄 Try Again"):
                model.predict(test_image)
    
    st.html("</div>")

st.html("""
<div class='footer'>
    <p>create by m.duy collaborator| Norttie </p>
</div>
""")