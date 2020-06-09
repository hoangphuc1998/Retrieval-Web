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
                $('#next').html('<p class="h3 text-center">Loading more...</p>')
                dataset = $('#dataset-combobox option:selected').val()
                $.get('http://128.199.238.213:8000/server/'+caption+'/'+dataset+'/cosine/10/'+start_from, function(ketqua) {
                //$.get('http://127.0.0.1:8000/server/'+caption+'/'+dataset+'/cosine/10/'+start_from, function(ketqua) {
                    let html_string=''
                    for (let i=0;i<ketqua.image.length;i++) {
                        html_string+='<div class="col-lg-4 col-md-12 mb-4">'
                        html_string+='<figure class="figure">'
                        html_string+='<img src="data:image/png;base64, ' +ketqua.image[i]+'" class="img-fluid mb-4" alt="">'
                        html_string+='<figcaption class="figure-caption">' +ketqua.filename[i]+'</figcaption>'
                        html_string+='</div>'

                    }
                    $('.row').append(html_string);
                    loading_content=false
                    start_from+=10
                    $('#next').html('')
                    console.log('done loading')
                });
        }
    }
});
