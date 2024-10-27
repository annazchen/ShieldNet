var file = new File([""], "filename");
$(document).ready(function(){
    $("#csv_input").change(function(){
        if(document.getElementById("csv_input").files.length == 0 ){
            document.getElementById("check_csv_input").innerHTML= "No files selected";
        }else{
            var filename = $('#csv_input').val().split('\\').pop();
            document.getElementById("check_csv_input").innerHTML= filename + " selected";
        }
    });
});

$(document).on("click", ":submit", function (e) {
    if (document.getElementById("csv_input").files.length == 0) {
        e.preventDefault();
        document.getElementById("check_csv_input").innerHTML= "Please upload a file!";
    }else{
        loading();
    }
});
function loading(){
    $("#logo-loading").show();
    $("#logo").hide();
}