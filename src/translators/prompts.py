def fixed_translationprompt(source_language,target_language,domain,input):
    translation_prompt = f"""This is an {source_language} to {target_language} translation in the field of {domain}, please provide the {
    target_language} translation for this text.\
    Do not provide any explanations or text apart from the translation. {source_language}: {input} {target_language}:"""
    
    return translation_prompt

def fixed_reflectionprompt(source_language,target_language,history,target_country):
    reflection_prompt = f"""Your task is to carefully read a content in the {history} and a translation from {source_language} to {target_language}, and then give constructive criticism and helpful suggestions to improve the translation. \
    The final style and tone of the translation should match the style of {target_language} colloquially spoken in {target_country}. When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
    (iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
    (iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_language}).
    Write a list of specific, helpful and constructive suggestions for improving the translation.
    Each suggestion should address one specific part of the translation.
    Output only the suggestions and nothing else."""
    
    return reflection_prompt

def fixed_editorprompt(source_language,target_language,history,suggestion):
    editor_prompt = f"""Your task is to carefully read, then edit, a translation of the content in the {history} from {source_language} to {target_language}, taking into\
    account a list of expert suggestions and constructive criticisms.

    // Expert Suggestions:
        {suggestion}

    Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:
    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions),
    (iii) style (by ensuring the translations reflect the style of the source text)
    (iv) terminology (inappropriate for context, inconsistent use), or
    (v) other errors.
            
    Output only the new translation and nothing else."""
    
    return editor_prompt

def orignal_translationprompt(source_language,target_language,input):
    orignaltranslation_prompt = f"""This is an {source_language} to {target_language} translation, please provide the {target_language} translation for this text. \
    Do not provide any explanations or text apart from the translation.
    {source_language}: {input}

    {target_language}:"""
    
    return orignaltranslation_prompt

def orignal_reflectionprompt(source_language,target_language,target_country,input,history):   

    orignalreflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_language} to {target_language}, and then give constructive criticism and helpful suggestions to improve the translation. \
    The final style and tone of the translation should match the style of {target_language} colloquially spoken in {target_country}.

    The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

    <SOURCE_TEXT>
    {input}
    </SOURCE_TEXT>

    <TRANSLATION>
    {history}
    </TRANSLATION>

    When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
    (iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
    (iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_language}).\n\

    Write a list of specific, helpful and constructive suggestions for improving the translation.
    Each suggestion should address one specific part of the translation.
    Output only the suggestions and nothing else."""
    
    return orignalreflection_prompt

def orignal_editorprompt(source_language,target_language,input,history,suggestion):

    orignaleditor_prompt = f"""Your task is to carefully read, then edit, a translation from {source_language} to {target_language}, taking into
    account a list of expert suggestions and constructive criticisms.

    The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
    as follows:

    <SOURCE_TEXT>
    {input}
    </SOURCE_TEXT>

    <TRANSLATION>
    {history}
    </TRANSLATION>

    <EXPERT_SUGGESTIONS>
    {suggestion}
    </EXPERT_SUGGESTIONS>

    Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

    (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
    (ii) fluency (by applying {target_language} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
    (iii) style (by ensuring the translations reflect the style of the source text)
    (iv) terminology (inappropriate for context, inconsistent use), or
    (v) other errors.

    Output only the new translation and nothing else."""
    
    return orignaleditor_prompt
