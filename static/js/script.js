let caption=$('#queryInput').val()

let dataset=$('#dataset-combobox option:selected').val()
window.onload = function() {
    var selItem = sessionStorage.getItem("SelItem");  
    $('#dataset-combobox').val(selItem);
    dataset = $('#dataset-combobox option:selected').val()
}
$('#dataset-combobox').change(function() { 
    var selVal = $(this).val();
    sessionStorage.setItem("SelItem", selVal);
});

let loading_content=false
let start_from = 10
$(window).scroll(function() {
    if (loading_content==false) {
        if($(window).scrollTop() >= $(document).height() - $(window).height()-30 && caption.length>0) {
                console.log('load more')
                loading_content=true
                dataset = $('#dataset-combobox option:selected').val()
                $.get('http://127.0.0.1:8000/server/'+caption+'/'+dataset+'/cosine/10/'+start_from, function(ketqua) {
                    let html_string=''
                    for (let i=0;i<ketqua.image.length;i++) {
                        html_string+='<div class="col-lg-4 col-md-12 mb-4"><img src="data:image/png;base64, ' +ketqua.image[i]+'" class="img-fluid mb-4" alt=""></div>'

                    }
                    $('.row').append(html_string);
                    loading_content=false
                    start_from+=10
                    console.log('done loading')
                });
        }
    }
});